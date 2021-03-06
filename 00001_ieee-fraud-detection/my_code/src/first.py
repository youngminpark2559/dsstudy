# conda activate py36_django_bare && \
# cd /mnt/external_disk/Companies/side_project/Kaggle/00001_ieee-fraud-detection/my_code/src && \
# rm e.l && python first.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
# v skewness

# data compress

# feature importance : xgb, lgbm, rf \cap PCA

# groupkfold evaluation

# Hyperparameter tuninig

# embedding approach for categorical features transformation

# - Ensemble
# - rankdata
# - sum predicted probabilities

# time, time delta, amount, browser, address

# v investigate "1 data" compare with 0 data

# drop useless columns 
# bi analysis, feature generate 

# outlier

# v impute nan

# categorycal data
#   - <25% appearing occurrence -> other category
#   - 140 absolute value threshold

# v if the ratio of nan is over 40%, that columns data is discarded

# I now understand why undersampling could be more convenient than oversampling
# undersampling is convenient for kfold evaluation

# This is the data which has time sequence information, so, I think shuffling should be avoided
# My strategy of generating train set 
# 0 class data (partial1) + 1 class data (entire)
# 0 class data (partial2) + 1 class data (entire)
# 0 class data (partial3) + 1 class data (entire)
# - in partial1, partial2, partial3, the time sequence should be kept
# - number of partial1 row == number of 1 class data row
# - last remain data is like 
#   - last_normal_data=normal_data.iloc[normal_data.shape[0]-714:,:]

# - data compressing was difficult
#   - when I process calculations like mean, std, normalizaton, type caused nan, inf

# experiment note
#  - managing skewness resulted in quite amount of difference in metric scores
#  - memory issue with oversampling, especially on hyperparameter tuning
#    - hyperparameter tuning (undersampling) / training (oversampling)
# - 1st test
#   - Private Score : 0.766517
#   - Public Score : 0.816813
#   - The following attempts
#     - I discarded features which have over 40% null
#     - Include more features
#     - Include identity data
#     - Use public notebook dataloading code
#     - null category : from mode to nullstr
#     - Finding importance features, and include only them
#     - Insert noise

# ================================================================================
import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn import init
from sklearn import preprocessing
from sklearn.utils import resample
import gc
import traceback
from collections import Counter,OrderedDict
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from hyperopt import hp,tpe,fmin
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split,cross_val_score,GroupKFold
import pickle
import copy
from datetime import datetime,timedelta,date
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.float_format','{:.2f}'.format)

# ================================================================================
# OPTIONS

# --------------------------------------------------------------------------------
# - Full train mode : 
USE_TRAIN=True ; TRAIN_DATA_SIZE="full" ; USE_TEST=False ; TEST_DATA_SIZE="small"
# - Fast train mode : 
# USE_TRAIN=True ; TRAIN_DATA_SIZE="small" ; USE_TEST=False ; TEST_DATA_SIZE="small"
# - Full test mode : 
# USE_TRAIN=False ; TRAIN_DATA_SIZE="small" ; USE_TEST=True ; TEST_DATA_SIZE="full"
# - Fast test mode : 
# USE_TRAIN=False ; TRAIN_DATA_SIZE="small" ; USE_TEST=True ; TEST_DATA_SIZE="small"
# - Full train and test mode : 
# USE_TRAIN=True ; TRAIN_DATA_SIZE="full" ; USE_TEST=True ; TEST_DATA_SIZE="full"
# - Fast train and test mode : 
# USE_TRAIN=True ; TRAIN_DATA_SIZE="small" ; USE_TEST=True ; TEST_DATA_SIZE="small"

# # USE_TRAIN=True
# USE_TRAIN=False

# TRAIN_DATA_SIZE="full"
# # TRAIN_DATA_SIZE="small"

# # USE_TEST=True
# USE_TEST=False

# # TEST_DATA_SIZE="full"
# TEST_DATA_SIZE="small"

USE_VALIDATION=True
# USE_VALIDATION=False

# --------------------------------------------------------------------------------
NAN_CRITERION=50
# CREATE_IMAGE_ON_NAN_RATIO=True
CREATE_IMAGE_ON_NAN_RATIO=False

# METHOD_FOR_IMPUTE_NUMERICAL_DATA="mice"
METHOD_FOR_IMPUTE_NUMERICAL_DATA="mean"

IMPUTED_NUMERICAL_DATA_SOURCE="function"
# IMPUTED_NUMERICAL_DATA_SOURCE="pickle"

ENCODED_CATEGORICAL_DATA_SOURCE="function"
# ENCODED_CATEGORICAL_DATA_SOURCE="pickle"

# RESAMPLING_USE=True
RESAMPLING_USE=False

# RESAMPLING="oversampling_smote"
RESAMPLING="oversampling_resampling"
# RESAMPLING="undersampling_resampling"

# HYPERPARAMETER_TUNIING_LGBM=True
HYPERPARAMETER_TUNIING_LGBM=False

# HYPERPARAMETER_TUNIING_LGBM_USE=True
HYPERPARAMETER_TUNIING_LGBM_USE=False

DEEP_LEARNING_EPOCH=1000

# ================================================================================
BATCH_SIZE=10000
# BATCH_SIZE=32000
# BATCH_SIZE=1000
# BATCH_SIZE=8
# number_of_features=165
number_of_features=187
number_of_labels=1
PATH="./trained_model.pth"

def perf_measure(y_actual,y_hat):
  TP=0
  FP=0
  TN=0
  FN=0

  for i in range(len(y_hat)): 
    if y_actual[i]==y_hat[i]==1:
      TP+=1
    if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
      FP+=1
    if y_actual[i]==y_hat[i]==0:
      TN+=1
    if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
      FN+=1

  return (TP,FP,TN,FN)

# ================================================================================
class Net(torch.nn.Module):
  def __init__(self,n_feature,n_hidden,n_output):
    super(Net,self).__init__()
    self.hidden=torch.nn.Linear(n_feature,n_hidden)
    self.hidden2=torch.nn.Linear(n_hidden,n_hidden)
    self.hidden3=torch.nn.Linear(n_hidden,n_hidden)
    self.hidden4=torch.nn.Linear(n_hidden,n_hidden)
    self.hidden5=torch.nn.Linear(n_hidden,n_hidden)
    self.predict=torch.nn.Linear(n_hidden,n_output)
    self.bn_after_hidden=nn.BatchNorm1d(num_features=n_hidden)
    self.bn_after_hidden2=nn.BatchNorm1d(num_features=n_hidden)
    self.bn_after_hidden3=nn.BatchNorm1d(num_features=n_hidden)
    self.bn_after_hidden4=nn.BatchNorm1d(num_features=n_hidden)
    self.bn_after_hidden5=nn.BatchNorm1d(num_features=n_hidden)
    self.dropout=torch.nn.Dropout(p=0.1)
    self.sigmoid=nn.Sigmoid()

    # ================================================================================
    # 모델의 파라미터 초기화

    for m in self.modules():
      if isinstance(m,nn.Conv2d):
        # Kaming Initialization
        init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)
      elif isinstance(m,nn.Linear):
        init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)

  def forward(self,x):
    after_hidden=self.hidden(x)
    after_hidden=self.bn_after_hidden(after_hidden)
    after_hidden=F.relu(after_hidden)
    after_hidden=self.dropout(after_hidden)

    after_hidden=self.hidden2(after_hidden)
    after_hidden=self.bn_after_hidden2(after_hidden)
    after_hidden=F.relu(after_hidden)
    after_hidden=self.dropout(after_hidden)

    after_hidden=self.hidden3(after_hidden)
    after_hidden=self.bn_after_hidden3(after_hidden)
    after_hidden=F.relu(after_hidden)
    after_hidden=self.dropout(after_hidden)

    after_hidden=self.hidden4(after_hidden)
    after_hidden=self.bn_after_hidden4(after_hidden)
    after_hidden=F.relu(after_hidden)
    after_hidden=self.dropout(after_hidden)

    after_hidden=self.hidden5(after_hidden)
    after_hidden=self.bn_after_hidden5(after_hidden)
    after_hidden=F.relu(after_hidden)
    after_hidden=self.dropout(after_hidden)

    x=self.predict(after_hidden)

    x=self.sigmoid(x)

    return x

class IEEE_Dataset(data.Dataset):
  def __init__(self,train_d):

    trainY=train_d["isFraud"]
    del train_d["isFraud"]

    self.train_X=np.array(train_d)
    self.train_y=np.array(trainY)
    # print("train_X",self.train_X.shape)
    # print("train_y",self.train_y.shape)
    # train_X (60253, 40)
    # train_y (60253, 6)

    self.number_of_data=self.train_X.shape[0]

  # ================================================================================
  def __len__(self):
    return self.number_of_data

  # ================================================================================
  def __getitem__(self,idx):
    return self.train_X[idx],self.train_y[idx]

class IEEEVal_Dataset(data.Dataset):
  def __init__(self,val_d):

    validationY=val_d["isFraud"]
    del val_d["isFraud"]

    self.test_X=np.array(val_d)
    self.test_y=np.array(validationY)
    # print("train_X",self.test_X.shape)
    # print("train_y",self.test_y.shape)

    self.number_of_data=self.test_X.shape[0]

  # ================================================================================
  def __len__(self):
    return self.number_of_data

  # ================================================================================
  def __getitem__(self,idx):
    return self.test_X[idx],self.test_y[idx]

# ================================================================================
def investigate_frequency(X_train,train_id):
  intersection_of_transaction_and_identity=set(list(X_train["TransactionID"])).intersection(set(list(train_id["TransactionID"])))
  # print('intersection_of_transaction_and_identity',intersection_of_transaction_and_identity)
  # print('intersection_of_transaction_and_identity',len(intersection_of_transaction_and_identity))

  transaction_minus_identity=set(list(X_train["TransactionID"]))-set(list(train_id["TransactionID"]))
  # print('transaction_minus_identity',transaction_minus_identity)
  # print('transaction_minus_identity',len(transaction_minus_identity))

  identity_minus_transaction=set(list(train_id["TransactionID"]))-set(list(X_train["TransactionID"]))
  # print('identity_minus_transaction',identity_minus_transaction)
  # print('identity_minus_transaction',len(identity_minus_transaction))
  # intersection_of_transaction_and_identity 144233
  # transaction_minus_identity 446307
  # identity_minus_transaction set()
  # identity_minus_transaction 0

  return len(intersection_of_transaction_and_identity),len(transaction_minus_identity),len(identity_minus_transaction)

def frequency_visualization(
  frequency_distribution_of_transaction_TransactionID,
  frequency_distribution_of_idendity_TransactionID,
  number_of_transaction_rows,
  number_of_identity_rows,
  intersection_of_transaction_and_identity,
  transaction_minus_identity,
  identity_minus_transaction):

  def my_fmt(x):
    return '{:.4f}%'.format(x)

  ax1=plt.subplot2grid((2,2),(0,0),colspan=1)
  ax2=plt.subplot2grid((2,2),(0,1),colspan=1)
  ax3=plt.subplot2grid((2,2),(1,0),colspan=2)
  ax1.pie(frequency_distribution_of_transaction_TransactionID.values(), labels=frequency_distribution_of_transaction_TransactionID.keys(),autopct=my_fmt)
  ax1.set_title('TransactionID distribution from transaction')
  ax2.pie(frequency_distribution_of_idendity_TransactionID.values(), labels=frequency_distribution_of_idendity_TransactionID.keys(),autopct=my_fmt)
  ax2.set_title('TransactionID distribution from identity')
  ax3.bar(["transaction","idendity","Transaction\capidentity","Transaction-Identity","Identity-Transaction"],[number_of_transaction_rows,number_of_identity_rows,intersection_of_transaction_and_identity,transaction_minus_identity,identity_minus_transaction])
  # for i, v in enumerate([number_of_transaction_rows,number_of_identity_rows,intersection_of_transaction_and_identity,transaction_minus_identity,identity_minus_transaction]):
  #   ax3.text(v + 3, i + .25, str(v), color='black', fontweight='bold')
  for index,data in enumerate([number_of_transaction_rows,number_of_identity_rows,intersection_of_transaction_and_identity,transaction_minus_identity,identity_minus_transaction]):
    plt.text(x=index,y=data+1,s=f"{data}",fontdict=dict(fontsize=15))
  # plt.close()
  # plt.show()

def visualize_nan_ratio_change_in_identity(
  train_id_shape,number_of_nan_in_column,ratio_of_nan_in_column,
  only_identity_shape,number_of_nan_in_column_after_merge,ratio_of_nan_in_column_after_merge,
  ratio_difference_of_nan_in_column_after_merge):
  # x : columns
  # y : number of NaNs
  # right y : NaN percent
  # line : NaN percent
  # group original/NaN 

  fig,ax=plt.subplots(2,1,figsize=(15,5))

  # --------------------------------------------------------------------------------
  labels=list(number_of_nan_in_column.index)
  # print('labels',labels)
  # ['TransactionID', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']
  data_number_of_row_in_column=[train_id_shape]*len(list(number_of_nan_in_column.values))
  data_number_of_nan_in_column=list(number_of_nan_in_column.values)
  data_ratio_of_nan_in_column=list(ratio_of_nan_in_column.values)

  width = 0.35
  x=np.arange(len(labels))
  
  rects1=ax[0].bar(x - width/2,data_number_of_row_in_column,width,label='Original identity rows')
  rects2=ax[0].bar(x + width/2,data_number_of_nan_in_column,width,label='Nan rows of original identity')

  ax0_right=ax[0].twinx()
  ax0_right.plot(x,list(ratio_of_nan_in_column.values))

  ax[0].set_ylabel('Number of rows')
  ax[0].set_title('Nan : Original identity')
  ax[0].set_xticks(x)
  ax[0].set_xticklabels(labels,rotation=90)
  ax[0].legend()

  # --------------------------------------------------------------------------------
  data_number_of_row_in_column_after_merge=[only_identity_shape]*len(list(number_of_nan_in_column.values))
  data_number_of_nan_in_column_after_merge=list(number_of_nan_in_column_after_merge.values)
  data_ratio_of_nan_in_column_after_merge=list(ratio_of_nan_in_column_after_merge.values)
  
  rects3=ax[1].bar(x-width/2,data_number_of_row_in_column_after_merge,width,label='Merged identity rows')
  rects4=ax[1].bar(x+width/2,data_number_of_nan_in_column_after_merge,width,label='Nan rows of Merged identity')

  ax1_right=ax[1].twinx()
  ax1_right.plot(x,list(ratio_of_nan_in_column_after_merge.values))

  ax[1].set_ylabel('Number of rows')
  ax[1].set_title('Nan : Merged identity')
  ax[1].set_xticks(x)
  ax[1].set_xticklabels(labels,rotation=90)
  ax[1].legend()

  fig.tight_layout()

  plt.show()

def nan_ratio_change_in_identity(train_id,X_train):
  number_of_nan_in_column=train_id.isnull().sum(axis=0)
  # print('number_of_nan_in_column',number_of_nan_in_column)
  # TransactionID         0
  # id_01                 0
  # id_02              3361
  # id_03             77909

  ratio_of_nan_in_column=number_of_nan_in_column/train_id.shape[0]*100
  # print('ratio_of_nan_in_column',ratio_of_nan_in_column)

  # ================================================================================
  X_train_only_identity=X_train[list(train_id.columns)]
  number_of_nan_in_column_after_merge=X_train_only_identity.isnull().sum(axis=0)
  # print('number_of_nan_in_column_after_merge',number_of_nan_in_column_after_merge)
  # number_of_nan_in_column_after_merge
  # TransactionID        0
  # id_01            15081
  # id_02            15194
  # id_03            17778
  # id_04            17778

  ratio_of_nan_in_column_after_merge=number_of_nan_in_column_after_merge/X_train_only_identity.shape[0]*100
  # print('ratio_of_nan_in_column_after_merge',ratio_of_nan_in_column_after_merge)
  # ratio_of_nan_in_column_after_merge
  # TransactionID     0.00
  # id_01             0.00
  # id_02            16.80
  # id_03           389.53

  # ================================================================================
  ratio_difference_of_nan_in_column_after_merge=ratio_of_nan_in_column_after_merge-ratio_of_nan_in_column

  # ================================================================================
  visualize_nan_ratio_change_in_identity(
    train_id.shape[0],number_of_nan_in_column,ratio_of_nan_in_column,
    X_train_only_identity.shape[0],number_of_nan_in_column_after_merge,ratio_of_nan_in_column_after_merge,
    ratio_difference_of_nan_in_column_after_merge)
  

def load_csv_files_train():
  
  # ================================================================================
  train_transaction_all_columns=['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181', 'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253', 'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321', 'V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339']
  train_identity_all_columns=['TransactionID', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

  # ================================================================================
  # Categorical columns in transaction

  categorical_columns_in_train_transaction=[]
  categorical_columns_in_train_transaction.append("ProductCD")
  categorical_columns_in_train_transaction.extend(['card{}'.format(i) for i in range(1,7)])
  categorical_columns_in_train_transaction.extend(['addr1','addr2'])
  categorical_columns_in_train_transaction.extend(['P_emaildomain','R_emaildomain'])
  categorical_columns_in_train_transaction.extend(['M{}'.format(i) for i in range(1,10)])

  numerical_columns_in_train_transaction=[one_column for one_column in train_transaction_all_columns if one_column not in categorical_columns_in_train_transaction]
  # print('numerical_columns_in_train_transaction',numerical_columns_in_train_transaction)
  # ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'dist1', 'dist2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 

  # ================================================================================
  # Categorical columns in identity

  categorical_columns_in_train_identity=[]
  categorical_columns_in_train_identity.extend(['DeviceType','DeviceInfo'])
  categorical_columns_in_train_identity.extend(['id_{}'.format(i) for i in range(12,39)])

  numerical_columns_in_train_identity=[one_column for one_column in train_identity_all_columns if one_column not in categorical_columns_in_train_identity]
  # print('numerical_columns_in_train_identity',numerical_columns_in_train_identity)
  # ['TransactionID', 'id_01', 'id_02', 'id_03', 'id_04',

  # ================================================================================
  dtypes = {}

  for c in numerical_columns_in_train_transaction+numerical_columns_in_train_identity:
    dtypes[c]='float32'
  for c in categorical_columns_in_train_transaction+categorical_columns_in_train_identity:
    dtypes[c]='category'
  # print('dtypes',dtypes)

  # ================================================================================
  # Load train data

  # print('dtypes',dtypes)

  if TRAIN_DATA_SIZE=="full":
    # X_train=pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/train_transaction_original.csv',dtype=dtypes,usecols=use_cols+['isFraud'])
    X_train=pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/train_transaction_original.csv', dtype=dtypes)

  elif TRAIN_DATA_SIZE=="small":

    # X_train=pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/train_transaction_small.csv',dtype=dtypes,usecols=use_cols+['isFraud'])
    X_train=pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/train_transaction_small.csv',dtype=dtypes)

  train_id=pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/train_identity.csv',dtype=dtypes)

  # ================================================================================
  # print(Counter(list(Counter(list(X_train["TransactionID"])).values())))
  # Counter({1: 20001})

  # print(Counter(list(Counter(list(train_id["TransactionID"])).values())))
  # Counter({1: 144233})

  # ================================================================================
  # intersection_of_transaction_and_identity,transaction_minus_identity,identity_minus_transaction=investigate_frequency(X_train,train_id)

  # frequency_visualization(
  #   Counter(list(Counter(list(X_train["TransactionID"])).values())),
  #   Counter(list(Counter(list(train_id["TransactionID"])).values())),
  #   X_train.shape[0],
  #   train_id.shape[0],
  #   intersection_of_transaction_and_identity,
  #   transaction_minus_identity,
  #   identity_minus_transaction)

  # pie : frequecy of TransactionID in transaction data   |   pie : frequecy of TransactionID in transaction data
  # vertical bar : number of transaction row, number of identity row, intersection row, transaction-identity row, identity-transaction row

  # ================================================================================
  # X_train = X_train.merge(train_id, how='left', left_index=True, right_index=True)
  X_train=pd.merge(X_train,train_id,on=['TransactionID'],how='left')

  X_train=X_train.sort_values(by=['TransactionID'],axis=0)
  # print('X_train',X_train)

  # ================================================================================
  # nan_ratio_change_in_identity(train_id,X_train)

  return X_train

def load_csv_files_test():

  # ================================================================================
  test_transaction_all_columns=['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181', 'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253', 'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321', 'V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339']
  test_identity_all_columns=['TransactionID', 'id-01', 'id-02', 'id-03', 'id-04', 'id-05', 'id-06', 'id-07', 'id-08', 'id-09', 'id-10', 'id-11', 'id-12', 'id-13', 'id-14', 'id-15', 'id-16', 'id-17', 'id-18', 'id-19', 'id-20', 'id-21', 'id-22', 'id-23', 'id-24', 'id-25', 'id-26', 'id-27', 'id-28', 'id-29', 'id-30', 'id-31', 'id-32', 'id-33', 'id-34', 'id-35', 'id-36', 'id-37', 'id-38', 'DeviceType', 'DeviceInfo']

  # ================================================================================
  # Categorical columns in transaction

  categorical_columns_in_test_transaction=[]
  categorical_columns_in_test_transaction.append("ProductCD")
  categorical_columns_in_test_transaction.extend(['card{}'.format(i) for i in range(1,7)])
  categorical_columns_in_test_transaction.extend(['addr1','addr2'])
  categorical_columns_in_test_transaction.extend(['P_emaildomain','R_emaildomain'])
  categorical_columns_in_test_transaction.extend(['M{}'.format(i) for i in range(1,10)])

  numerical_columns_in_test_transaction=[one_column for one_column in test_transaction_all_columns if one_column not in categorical_columns_in_test_transaction]
  # print('numerical_columns_in_test_transaction',numerical_columns_in_test_transaction)
  # ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'dist1', 'dist2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 

  # ================================================================================
  # Categorical columns in identity

  categorical_columns_in_test_identity=[]
  categorical_columns_in_test_identity.extend(['DeviceType','DeviceInfo'])
  categorical_columns_in_test_identity.extend(['id-{}'.format(i) for i in range(12,39)])

  numerical_columns_in_test_identity=[one_column for one_column in test_identity_all_columns if one_column not in categorical_columns_in_test_identity]
  # print('numerical_columns_in_test_identity',numerical_columns_in_test_identity)
  # ['TransactionID', 'id_01', 'id_02', 'id_03', 'id_04',

  # ================================================================================
  dtypes = {}

  for c in numerical_columns_in_test_transaction+numerical_columns_in_test_identity:
    dtypes[c]='float32'
  for c in categorical_columns_in_test_transaction+categorical_columns_in_test_identity:
    dtypes[c]='category'
  # print('dtypes',dtypes)

  # ================================================================================
  # Load train data

  # print('dtypes',dtypes)

  if TEST_DATA_SIZE=="full":
    # X_train=pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/train_transaction_original.csv',dtype=dtypes,usecols=use_cols+['isFraud'])
    X_train=pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/test_transaction_original.csv',dtype=dtypes,)

  elif TEST_DATA_SIZE=="small":

    # X_train=pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/train_transaction_small.csv',dtype=dtypes,usecols=use_cols+['isFraud'])
    X_train=pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/test_transaction.csv',dtype=dtypes)

  train_id=pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/test_identity.csv',dtype=dtypes)

  # ================================================================================
  # X_train=X_train.merge(train_id,how='left',left_index=True,right_index=True)
  X_train=pd.merge(X_train,train_id,on=['TransactionID'],how='left')

  X_train=X_train.sort_values(by=['TransactionID'],axis=0)
  # print('X_train',X_train)

  return X_train

def check_nan(merged,NAN_CRITERION,create_image):
  number_of_rows_from_data=merged.shape[0]
  number_of_columns_from_data=merged.shape[1]

  # ================================================================================
  number_of_nan_in_column=merged.isnull().sum(axis=0)
  number_of_nan_in_row=merged.isnull().sum(axis=1)
  # print("number_of_nan_in_column",number_of_nan_in_column)
  # print("number_of_nan_in_row",number_of_nan_in_row)

  # ================================================================================
  # with pd.option_context('display.max_rows',100000):
  #   print("number_of_nan_in_column/number_of_rows_from_data*100",number_of_nan_in_column/number_of_rows_from_data*100)
  
  # with pd.option_context('display.max_rows',100000):
  #   print("number_of_nan_in_row/number_of_columns_from_data*100",number_of_nan_in_row/number_of_columns_from_data*100)

  # ================================================================================
  # Create dataframe

  df=(number_of_nan_in_column/number_of_rows_from_data*100).to_frame().reset_index()
  # print('df',df)
  #               index          0
  # 0     TransactionID   0.000000
  # 1           isFraud   0.000000
  # 2     TransactionDT   0.000000
  # 3    TransactionAmt   0.000000
  # 4         ProductCD   0.000000
  # ..              ...        ...
  # 429           id_36  76.126088
  # 430           id_37  76.126088
  # 431           id_38  76.126088
  # 432      DeviceType  76.155722
  # 433      DeviceInfo  79.905510

  # [434 rows x 2 columns]

  # ================================================================================
  df=df.rename(columns={"index":'column_name',0:'nan_percent'})

  fn='./results_csv/nan_percent.csv'
  df.to_csv(fn,sep=',',encoding='utf-8',index=False)

  # ================================================================================
  columns_to_be_dropped=list((df[df['nan_percent']>NAN_CRITERION])['column_name'])
  # print('columns_to_be_dropped',columns_to_be_dropped)
  # ['dist1', 'dist2', 'R_emaildomain', 'D5', 'D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14', 'M5', 'M7', 'M8', 'M9', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181', 'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253', 'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278', 'V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

  # ================================================================================
  if create_image==True:
    plt.bar(list(df["column_name"]),list(df["nan_percent"]))
    plt.xticks(rotation=90,fontsize=0.3)
    # plt.show()
    plt.savefig('NaN_distribution.png',dpi=4000)
    # /mnt/external_disk/Companies/side_project/Kaggle/00001_ieee-fraud-detection/my_code/src/aa.png
  
  return df,columns_to_be_dropped

def discard_nan_columns(merged,columns_to_be_dropped):
  # print('merged',merged)
  #        TransactionID  isFraud  TransactionDT  TransactionAmt ProductCD  card1  \
  # 13285     2987031.00     0.00       86998.00          363.89         W   6573   
  # 483       2987111.00     0.00       88383.00           18.19         C  13832   
  # 6239      2987261.00     0.00       90492.00           59.00         W  10049   

  # ================================================================================
  merged.drop(columns_to_be_dropped,axis=1,inplace=True)
 
  return merged

def separate_full_column_data_into_categorical_and_numerical(csv_train):

  # with pd.option_context('display.max_rows',100000):
  #   print('csv_train',csv_train.dtypes)

  # ================================================================================
  # Set index

  csv_train=csv_train.set_index("TransactionID")
  # print('csv_train',csv_train)

  # ================================================================================
  # For train data

  numerical_data=[]
  categorical_data=[]
  for one_column_name in csv_train:
    # print('one_column_name',one_column_name)
    # one_column_name TransactionID

    # print("csv_train[one_column_name].dtype",csv_train[one_column_name])
    # float32

    # print('str(csv_train[one_column_name].dtype)',str(csv_train[one_column_name].dtype))
    
    if 'float' in str(csv_train[one_column_name].dtype) or 'int' in str(csv_train[one_column_name].dtype):
      # print('csv_train[[one_column_name]]',csv_train[[one_column_name]])
      numerical_data.append(csv_train[[one_column_name]])
    else:
      # print('csv_train[[one_column_name]]',csv_train[[one_column_name]])
      categorical_data.append(csv_train[[one_column_name]])

  numerical_train_df=pd.concat(numerical_data,axis=1)
  categorical_train_df=pd.concat(categorical_data,axis=1)
  # print("numerical_df",numerical_df)
  # print("categorical_df",categorical_df)

  return numerical_train_df,categorical_train_df

def impute_categorical_data_by_mode(under40_nan_categorical_df):

  # ================================================================================
  # Find mode (train)

  temp_df1=[]
  # under40_nan_categorical_df_mode=under40_nan_categorical_df.mode().astype(str)
  for one_column_name in under40_nan_categorical_df:
    

    # one_column_data=under40_nan_categorical_df[[one_column_name]]
    # bb=under40_nan_categorical_df[[one_column_name]].fillna(one_column_data.value_counts().index[0][0])

    under40_nan_categorical_df[one_column_name]=under40_nan_categorical_df[one_column_name].cat.add_categories('nullstr')
    bb=under40_nan_categorical_df[[one_column_name]].fillna("nullstr")

    temp_df1.append(bb)
  
  temp_train_df2=pd.concat(temp_df1,axis=1)
  # print('temp_df2',temp_df2)

  return temp_train_df2

def impute_numerical_data_by_MICE(under40_nan_numerical_df):
  # print('under40_nan_numerical_df',under40_nan_numerical_df)

  # numerical_df_MiceImputed=under40_nan_numerical_df.copy(deep=True) 
  mice_imputer=IterativeImputer()
  under40_nan_numerical_df.iloc[:,:]=mice_imputer.fit_transform(under40_nan_numerical_df)
  # print('numerical_df_MiceImputed',numerical_df_MiceImputed)

  number_of_nan_in_entire_columns=under40_nan_numerical_df.isnull().sum(axis=0).sum()
  # print('number_of_nan_in_entire_columns',number_of_nan_in_entire_columns)

  assert number_of_nan_in_entire_columns==0,'number_of_nan_in_entire_columns!=0'

  with open('./pickles/imputed_numerical_df.pkl','wb') as f:
    pickle.dump(under40_nan_numerical_df,f)
  
  return under40_nan_numerical_df

def encode_categorical_data_using_LabelEncoder(imputed_categorical_df,datatype):
  # print('imputed_categorical_df',imputed_categorical_df)

  # ================================================================================
  # Perform label encoding

  label_encoders = {}
  for one_column_name in list(imputed_categorical_df.columns):
    label_encoders[one_column_name]=LabelEncoder()

    imputed_categorical_df[one_column_name]=label_encoders[one_column_name].fit_transform(imputed_categorical_df[one_column_name].astype(str))
  
  # ================================================================================
  # print('imputed_categorical_df',imputed_categorical_df)
  #                ProductCD  card1  card2  card3  card5  card6  addr1  addr2  \
  # TransactionID                                                               
  # 2987031.00             4     27     56      6     18      0     27     17   
  # 2987111.00             0     27     28      7     17      1     51     20   
  # 2987261.00             4     27     52      6     18      1     10     17   
  # 2987274.00             4      1     25      6     10      1     50     17   
  # 2987285.00             4     27     49      6     17      1     24     17   
  # ...                  ...    ...    ...    ...    ...    ...    ...    ...   
  # 3577426.00             4     27     38      6      2      1     21     17   
  # 3577436.00             4     27     52      6      2      1     21     17   
  # 3577443.00             4     27     58      6     18      1     33     17   
  # 3577451.00             2     27     32      6      7      0     24     17   
  # 3577466.00             4     27      6      6      2      1      2     17   

  #               P_emaildomain  R_emaildomain  M1  M2  M3  M4  M6  M7  M8  M9  \
  # TransactionID                                                                 
  # 2987031.00                51             31   1   1   1   3   1   0   1   1   
  # 2987111.00                19             17   2   2   2   0   2   2   2   2   
  # 2987261.00                34             31   1   1   1   1   1   2   2   2   
  # 2987274.00                19             31   2   2   2   0   0   2   2   2   
  # 2987285.00                16             31   1   1   1   0   1   0   1   1   
  # ...                      ...            ...  ..  ..  ..  ..  ..  ..  ..  ..   
  # 3577426.00                16             31   1   1   1   3   0   0   1   1   
  # 3577436.00                16             31   1   1   1   0   0   2   2   2   
  # 3577443.00                 2             31   1   1   1   3   0   0   0   1   
  # 3577451.00                16             14   2   2   2   3   2   2   2   2   
  # 3577466.00                51             31   1   0   0   0   1   0   0   0   

  #               id_12  id_15  id_16  id_28  id_29  id_31  id_35  id_36  id_37  \
  # TransactionID                                                                  
  # 2987031.00         2      3      2      2      2     35      2      2      2   
  # 2987111.00         1      1      1      0      0      7      0      0      1   
  # 2987261.00         2      3      2      2      2     35      2      2      2   
  # 2987274.00         2      3      2      2      2     35      2      2      2   
  # 2987285.00         2      3      2      2      2     35      2      2      2   
  # ...              ...    ...    ...    ...    ...    ...    ...    ...    ...   
  # 3577426.00         2      3      2      2      2     35      2      2      2   
  # 3577436.00         2      3      2      2      2     35      2      2      2   
  # 3577443.00         2      3      2      2      2     35      2      2      2   
  # 3577451.00         1      1      1      1      1     36      1      0      1   
  # 3577466.00         2      3      2      2      2     35      2      2      2   

  #               id_38  DeviceType  DeviceInfo  card1_addr1  \
  # TransactionID                                               
  # 2987031.00         2           2           4          610   
  # 2987111.00         1           0           2          634   
  # 2987261.00         2           2           4          593   
  # 2987274.00         2           2           4           36   
  # 2987285.00         2           2           4          607   
  # ...              ...         ...         ...          ...   
  # 3577426.00         2           2           4          604   
  # 3577436.00         2           2           4          604   
  # 3577443.00         2           2           4          616   
  # 3577451.00         0           1           3          607   
  # 3577466.00         2           2           4          585   

  #               card1_addr1_P_emaildomain  
  # TransactionID                             
  # 2987031.00                          2083  
  # 2987111.00                          2382  
  # 2987261.00                          1829  
  # 2987274.00                           102  
  # 2987285.00                          2027  
  # ...                                  ...  
  # 3577426.00                          1972  
  # 3577436.00                          1972  
  # 3577443.00                          2168  
  # 3577451.00                          2027  
  # 3577466.00                          1725  

  # [20001 rows x 32 columns]

  # ================================================================================
  if datatype=="train":
    with open('./pickles/imputed_categorical_train_df.pkl','wb') as f:
      pickle.dump(imputed_categorical_df,f)
  elif datatype=="test":
    with open('./pickles/imputed_categorical_test_df.pkl','wb') as f:
      pickle.dump(imputed_categorical_df,f)

  return imputed_categorical_df

def concat_numerical_and_categorical_data(imputed_numerical_df,encoded_categorical_data):
  concated_data=pd.merge(imputed_numerical_df,encoded_categorical_data,on=['TransactionID'],how='left')
  return concated_data

def perform_normalization(train_X):
  train_y=train_X[["isFraud"]]
  DT_M=train_X[["DT_M"]]
  del train_X["isFraud"]
  del train_X["DT_M"]

  # print('train_X.mean()',round(train_X.mean()))
  # print('train_X.std()',train_X.std())
  normalized_train_X=(train_X-train_X.mean())/(train_X.std()+1e-6)
  # print('normalized_df',normalized_df)

  normalized_train_X=pd.merge(normalized_train_X,DT_M,on=['TransactionID'],how='left')
  # print('normalized_train_X',normalized_train_X)

  return normalized_train_X,train_y

def evaluate_by_lgbm(normalized_train_X,train_y,best_parameter_dict=None,model_save=False):
  
  group_kfold=GroupKFold(n_splits=3)
  groups=list(normalized_train_X['DT_M'])

  for fold_n,(train,test) in enumerate(group_kfold.split(normalized_train_X,train_y,groups)):
  
    # print('train',train.shape)
    # print('test',test.shape)
    # train (916,)
    # test (512,)
    # train (974,)
    # test (454,)
    # train (966,)
    # test (462,)

    print(fold_n)
        
    X_train_,X_valid=normalized_train_X.iloc[train],normalized_train_X.iloc[test]
    y_train_,y_valid=train_y.iloc[train],train_y.iloc[test]


    # # ================================================================================
    # # Prepare train and validation datasets

    # X_train_,X_valid=normalized_train_X,full_validation_X
    # y_train_,y_valid=smote_train_y,full_validation_y

    # del X_valid["TransactionID"]

    # ================================================================================
    # Convert data into lgb data

    # dtrain=lgb.Dataset(X_train,label=y_train)
    # dvalid=lgb.Dataset(X_valid,label=y_valid)

    # ================================================================================
    if best_parameter_dict!=None:

      params={
        'num_leaves':int(best_parameter_dict['num_leaves']),
        'n_estimators':int(best_parameter_dict['n_estimators']),
        'max_depth':int(best_parameter_dict['max_depth']),
        'subsample_for_bin':int(best_parameter_dict['subsample_for_bin']),
        'learning_rate':best_parameter_dict['learning_rate'],
        'subsample':best_parameter_dict['subsample'],
        'colsample_bytree':best_parameter_dict['colsample_bytree'],
        'min_child_samples':int(best_parameter_dict['min_child_samples']),
        'min_child_weight':best_parameter_dict['min_child_weight'],
        'min_split_gain':best_parameter_dict['min_split_gain'],
        'reg_lambda':best_parameter_dict['reg_lambda'],
        #'reg_alpha':best_parameter_dict['reg_alpha']}
      }

      # ================================================================================
      # create lgbm model with params

      lgbclf=lgb.LGBMClassifier(**params)

    else:
      params={
        'num_leaves':512,
        'n_estimators':512,
        'max_depth':9,
        'learning_rate':0.064,
        'subsample':0.85,
        'colsample_bytree':0.85,
        'boosting_type':'gbdt',
        'reg_alpha':0.3
      }

      # params={
      #   "num_leaves":512,
      #   "n_estimators":512,
      #   "max_depth":9,
      #   "learning_rate":0.064,
      #   "subsample":0.85,
      #   "colsample_bytree":0.85,
      #   "boosting_type":"gbdt",
      #   "reg_alpha":0.3,
      #   "reg_lamdba":0.243,
      #   "metric":"AUC"
      # }

      # params = {
      #   'num_leaves': 491,
      #   'min_child_weight': 0.03454472573214212,
      #   'feature_fraction': 0.3797454081646243,
      #   'bagging_fraction': 0.4181193142567742,
      #   'min_data_in_leaf': 106,
      #   'objective': 'binary',
      #   'max_depth': -1,
      #   'learning_rate': 0.006883242363721497,
      #   "boosting_type": "gbdt",
      #   "bagging_seed": 11,
      #   "metric": 'auc',
      #   "verbosity": -1,
      #   'reg_alpha': 0.3899927210061127,
      #   'reg_lambda': 0.6485237330340494,
      #   'random_state': 47
      # }

      lgbclf=lgb.LGBMClassifier(**params)

    # ================================================================================
    # Train lgb model with train dataset

    lgbclf.fit(X_train_,y_train_.values.ravel())
    
    # ================================================================================
    # Delete used data

    del X_train_,y_train_

    # ================================================================================
    # Make prediction on test dataset

    val=lgbclf.predict_proba(X_valid)[:,1]
    # print("pred",pred)

    # ================================================================================
    # Delete used data

    del X_valid

    # ================================================================================
    # from scipy.stats import rankdata, spearmanr
    # from sklearn.metrics import roc_auc_score

    print('ROC accuracy: {}'.format(roc_auc_score(y_valid,val)))
    
    from sklearn.metrics import accuracy_score
    print("accuracy_score",accuracy_score(y_valid,val.round()))

    from sklearn.metrics import confusion_matrix
    print("TN FP\nFN TP\n",confusion_matrix(y_valid,val.round()))

    from sklearn.metrics import f1_score
    print("f1_score average='macro'",f1_score(y_valid,val.round(),average='macro'))
    print("f1_score average='micro'",f1_score(y_valid,val.round(),average='micro'))
    print("f1_score average='weighted'",f1_score(y_valid,val.round(),average='weighted'))
    print("f1_score average='None'",f1_score(y_valid,val.round(),average=None))
    print("f1_score zero_division=1",f1_score(y_valid,val.round(),zero_division=1))

    from sklearn.metrics import recall_score
    print("recall_score average='macro'",recall_score(y_valid,val.round(),average='macro'))
    print("recall_score average='micro'",recall_score(y_valid,val.round(),average='micro'))
    print("recall_score average='weighted'",recall_score(y_valid,val.round(),average='weighted'))
    print("recall_score average='None'",recall_score(y_valid,val.round(),average=None))
    print("recall_score zero_division=1",recall_score(y_valid,val.round(),zero_division=1))

    from sklearn.metrics import precision_score
    print("precision_score average='macro'",precision_score(y_valid,val.round(),average='macro'))
    print("precision_score average='micro'",precision_score(y_valid,val.round(),average='micro'))
    print("precision_score average='weighted'",precision_score(y_valid,val.round(),average='weighted'))
    print("precision_score average='None'",precision_score(y_valid,val.round(),average=None))
    print("precision_score zero_division=1",precision_score(y_valid,val.round(),zero_division=1))

    # ================================================================================
    del val,y_valid

    if model_save==True:
      with open('./pickles/trained_model.pkl','wb') as f:
        pickle.dump(lgbclf,f)

  return lgbclf

def optimize_hyperparameter_of_lgbm(normalized_train_X,smote_train_y):

  # del normalized_train_X["TransactionID"]
 
  # ================================================================================
  # Create LightGBM model with parameter placeholder

  def objective(params):
      params={
        'n_estimators':int(params['n_estimators']),
        'subsample_for_bin':int(params['subsample_for_bin']),
        'learning_rate':params['learning_rate'],
        'max_depth':int(params['max_depth']),
        'num_leaves':int(params['num_leaves']),
        'subsample':params['subsample'],
        'colsample_bytree':params['colsample_bytree'],
        'min_child_samples':int(params['min_child_samples']),
        'min_child_weight':params['min_child_weight'],
        'min_split_gain':params['min_split_gain'],
        'reg_lambda':params['reg_lambda'],
        #'reg_alpha':params['reg_alpha']}
      }

      # ================================================================================
      # create lgbm model with params

      lgb_a=lgb.LGBMClassifier(**params)
      score=cross_val_score(lgb_a,normalized_train_X,smote_train_y,cv=5,n_jobs=-1).mean()
      print('score',score)
      return score

  # ================================================================================
  # Validation parameter grid

  valgrid={
    'n_estimators':hp.quniform('n_estimators',1000,5000,50),
    'subsample_for_bin':hp.uniform('subsample_for_bin',10,300000),
    'learning_rate':hp.uniform('learning_rate',0.00001,0.03),
    'max_depth':hp.quniform('max_depth',3,8,1),
    'num_leaves':hp.quniform('num_leaves',7,256,1),
    'subsample':hp.uniform('subsample',0.60,0.95),
    'colsample_bytree':hp.uniform('colsample_bytree',0.60,0.95),
    'min_child_samples':hp.quniform('min_child_samples',1,500,1),
    'min_child_weight':hp.uniform('min_child_weight',0.60,0.95),
    'min_split_gain':hp.uniform('min_split_gain',0.60,0.95), 
    'reg_lambda':hp.uniform('reg_lambda',1,25)
    #'reg_alpha':hp.uniform('reg_alpha',1,25)  
  }

  # ================================================================================
  bestP=fmin(fn=objective,space=valgrid,max_evals=20,rstate=np.random.RandomState(123),algo=tpe.suggest)
  # print('bestP',bestP)

  with open('./pickles/best_parameter_dict.pkl','wb') as f:
    pickle.dump(bestP,f)
  
def split_train_and_validation_before_oversampling(concated_numerical_categorical_data):
  # print('concated_numerical_categorical_data',concated_numerical_categorical_data)
  
  concated_numerical_categorical_data=shuffle(concated_numerical_categorical_data)

  # ================================================================================
  full_normal=concated_numerical_categorical_data[concated_numerical_categorical_data['isFraud']==0]
  full_fraud=concated_numerical_categorical_data[concated_numerical_categorical_data['isFraud']==1]
  # print("full_normal",full_normal)
  # print("full_fraud",full_fraud)

  # ================================================================================
  full_normal_y=full_normal[['isFraud']]
  full_fraud_y=full_fraud[['isFraud']]
  
  del full_normal['isFraud']
  del full_fraud['isFraud']

  # ================================================================================
  normal_train_X,normal_validation_X,normal_train_y,normal_validation_y=train_test_split(full_normal,full_normal_y,test_size=.2,random_state=1)
  fraud_train_X,fraud_validation_X,fraud_train_y,fraud_validation_y=train_test_split(full_fraud,full_fraud_y,test_size=.2,random_state=1)

  full_train_X=pd.concat([normal_train_X,fraud_train_X])
  full_train_y=pd.concat([normal_train_y,fraud_train_y])
  full_validation_X=pd.concat([normal_validation_X,fraud_validation_X])
  full_validation_y=pd.concat([normal_validation_y,fraud_validation_y])

  # print("full_train_X@",full_train_X)
  # print("full_train_y@",full_train_y)
  # print("full_validation_X@",full_validation_X)
  # print("full_validation_y@",full_validation_y)

  return full_train_X,full_train_y,full_validation_X,full_validation_y

def perform_oversampling(full_train_X,full_train_y):
  smote=SMOTE()
  smote_train_X,smote_train_y=smote.fit_sample(full_train_X,full_train_y)
  # print("X_sm",X_sm.shape)
  # print("y_sm",y_sm.shape)
  # X_sm (30858, 201)
  # y_sm (30858, 1)
  
  return smote_train_X,smote_train_y

def reduce_mem_usage(df):
    
  # print(df.memory_usage())
  # Index             407688
  # isFraud            80008
  # TransactionDT      80008
  # TransactionAmt     80008
  # ProductCD          80008
  #                 ...  
  # id_36              80008
  # id_37              80008
  # id_38              80008
  # DeviceType         80008
  # DeviceInfo         80008
  # Length: 434, dtype: int64

  start_mem=df.memory_usage().sum()/1024**2
  # print("start_mem",start_mem)
  # start_mem 33.42738342285156
  print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
  # Memory usage of dataframe is 33.43 MB

  # ================================================================================
  for col in df.columns:
    # print("col",col)
    # col isFraud

    col_type=df[col].dtype
    # print("col_type",col_type)
    # col_type int64
    
    # object type has no min and max values 
    if col_type!=object:
      c_min=df[col].min()
      c_max=df[col].max()
      # print("c_min",c_min)
      # print("c_max",c_max)
      # c_min 0
      # c_max 1

      # if column_type is int
      if str(col_type)[:3]=='int':

        # print("np.iinfo(np.int8).min",np.iinfo(np.int8).min)
        # print("np.iinfo(np.int8).max",np.iinfo(np.int8).max)
        # print("np.iinfo(np.int16).min",np.iinfo(np.int16).min)
        # print("np.iinfo(np.int16).max",np.iinfo(np.int16).max)
        # print("np.iinfo(np.int32).min",np.iinfo(np.int32).min)
        # print("np.iinfo(np.int32).max",np.iinfo(np.int32).max)
        # print("np.iinfo(np.int64).min",np.iinfo(np.int64).min)
        # print("np.iinfo(np.int64).max",np.iinfo(np.int64).max)
        # np.iinfo(np.int8).min -128
        # np.iinfo(np.int8).max 127
        # np.iinfo(np.int16).min -32768
        # np.iinfo(np.int16).max 32767
        # np.iinfo(np.int32).min -2147483648
        # np.iinfo(np.int32).max 2147483647
        # np.iinfo(np.int64).min -9223372036854775808
        # np.iinfo(np.int64).max 9223372036854775807

        if c_min>np.iinfo(np.int8).min and c_max<np.iinfo(np.int8).max:
          df[col]=df[col].astype(np.int8)
        elif c_min>np.iinfo(np.int16).min and c_max<np.iinfo(np.int16).max:
          df[col]=df[col].astype(np.int16)
        elif c_min>np.iinfo(np.int32).min and c_max<np.iinfo(np.int32).max:
          df[col]=df[col].astype(np.int32)
        elif c_min>np.iinfo(np.int64).min and c_max<np.iinfo(np.int64).max:
          df[col]=df[col].astype(np.int64)  
      else:
        # print("np.finfo(np.float16).min",np.finfo(np.float16).min)
        # print("np.finfo(np.float16).max",np.finfo(np.float16).max)
        # print("np.finfo(np.float32).min",np.finfo(np.float32).min)
        # print("np.finfo(np.float32).max",np.finfo(np.float32).max)
        # np.finfo(np.float16).min -65500.0
        # np.finfo(np.float16).max 65500.0
        # np.finfo(np.float32).min -3.4028235e+38
        # np.finfo(np.float32).max 3.4028235e+38

        if c_min>np.finfo(np.float16).min and c_max<np.finfo(np.float16).max:
          df[col]=df[col].astype(np.float16)
        elif c_min>np.finfo(np.float32).min and c_max<np.finfo(np.float32).max:
          df[col]=df[col].astype(np.float32)
        else:
          df[col]=df[col].astype(np.float64)
    
    # when the column is object type
    else:
      # df[col]=df[col].astype('category')
      continue

  end_mem=df.memory_usage().sum()/1024**2
  print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
  print('Decreased by {:.1f}%'.format(100*(start_mem-end_mem)/start_mem))
  # Memory usage after optimization is: 8.90 MB
  # Decreased by 73.4%
  
  return df

def impute_numerical_data_by_mean(under40_nan_numerical_df,datatype):
  # print('under40_nan_numerical_df',under40_nan_numerical_df)
  
  # under40_nan_numerical_df.fillna(under40_nan_numerical_df.mean(),inplace=True)

  # under40_nan_numerical_df.apply(lambda x:x.fillna(x.mean(),inplace=True))
  # print('imputed',imputed.dtypes)
  # print('imputed',imputed)
  # print('imputed',imputed.isnull().sum(axis=0))

  temp_df=[]
  for one_column_name in under40_nan_numerical_df:

    one_df=under40_nan_numerical_df[[one_column_name]]
    one_df_mean=one_df.mean()
    temp_df.append(one_df.fillna(one_df_mean))
  
  temp_df2=pd.concat(temp_df,axis=1)
  # print('temp_df2',temp_df2)

  # ================================================================================
  number_of_nan_in_entire_columns=temp_df2.isnull().sum(axis=0).sum()
  assert number_of_nan_in_entire_columns==0,'number_of_nan_in_entire_columns!=0'

  if datatype=="train":
    with open('./pickles/imputed_numerical_train_df.pkl','wb') as f:
      pickle.dump(temp_df2,f)
  elif datatype=="test":
    with open('./pickles/imputed_numerical_test_df.pkl','wb') as f:
      pickle.dump(temp_df2,f)

  return temp_df2

def perform_oversampling_resampling(full_train_X,full_train_y):

  X=pd.concat([full_train_X,full_train_y],axis=1)
  # print('X',X)
  
  # ================================================================================
  # Separate data into 2 groups (fraud, not fraud)
  
  not_fraud=X[X.isFraud==0]
  fraud=X[X.isFraud==1]
  
  # ================================================================================
  # upsample minority class data
  
  fraud_upsampled=resample(
    fraud,
    replace=True, # sample with replacement
    n_samples=len(not_fraud), # match number in majority class
    random_state=27) # reproducible results
  
  # ================================================================================
  # combine majority and upsampled minority
  
  upsampled=pd.concat([not_fraud,fraud_upsampled])
  
  # ================================================================================
  train_y=upsampled["isFraud"]
  del upsampled["isFraud"]

  return upsampled,train_y

def perform_undersampling_resampling(full_train_X,full_train_y):
  
  X=pd.concat([full_train_X,full_train_y],axis=1)
  # print('X',X)
  
  # ================================================================================
  # Separate data into 2 groups (fraud, not fraud)
  
  not_fraud=X[X.isFraud==0]
  fraud=X[X.isFraud==1]
  
  # ================================================================================
  # undersampling
  not_fraud_downsampled=resample(
    not_fraud,
    replace=False, # sample without replacement
    n_samples=len(fraud), # match minority n
    random_state=27) # reproducible results

  # ================================================================================
  # combine majority and upsampled minority
  
  downsampled=pd.concat([not_fraud_downsampled,fraud])
  
  # ================================================================================
  train_y=downsampled["isFraud"]
  del downsampled["isFraud"]

  return downsampled,train_y

def manage_skewness(imputed_numerical_df):
  # print('imputed_numerical_df',imputed_numerical_df.skew())
  # isFraud             5.061223
  # TransactionDT       0.131155
  # TransactionAmt     14.374490
  # C1                 23.957960
  # C2                 23.677433
  #                     ...    
  # V317               24.597657
  # V318               25.483088
  # V319              181.835501
  # V320               70.516271
  # V321              123.557730
  # Length: 190, dtype: float64

  # ================================================================================
  imputed_numerical_df_columns=list(imputed_numerical_df.columns)
  # print('imputed_numerical_df_columns',imputed_numerical_df_columns)
  # imputed_numerical_df_columns ['TransactionID', 'TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'dist1', 'dist2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'V1', 'V3', 'V4', 'V6', 'V8', 'V11', 'V13', 'V14', 'V17', 'V20', 'V23', 'V26', 'V27', 'V30', 'V36', 'V37', 'V40', 'V41', 'V44', 'V47', 'V48', 'V54', 'V56', 'V59', 'V62', 'V65', 'V67', 'V68', 'V70', 'V76', 'V78', 'V80', 'V82', 'V86', 'V88', 'V89', 'V91', 'V107', 'V108', 'V111', 'V115', 'V117', 'V120', 'V121', 'V123', 'V124', 'V127', 'V129', 'V130', 'V136', 'V138', 'V139', 'V142', 'V147', 'V156', 'V160', 'V162', 'V165', 'V166', 'V169', 'V171', 'V173', 'V175', 'V176', 'V178', 'V180', 'V182', 'V185', 'V187', 'V188', 'V198', 'V203', 'V205', 'V207', 'V209', 'V210', 'V215', 'V218', 'V220', 'V221', 'V223', 'V224', 'V226', 'V228', 'V229', 'V234', 'V235', 'V238', 'V240', 'V250', 'V252', 'V253', 'V257', 'V258', 'V260', 'V261', 'V264', 'V266', 'V267', 'V271', 'V274', 'V277', 'V281', 'V283', 'V284', 'V285', 'V286', 'V289', 'V291', 'V294', 'V296', 'V297', 'V301', 'V303', 'V305', 'V307', 'V309', 'V310', 'V314', 'V320', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_13', 'id_14', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_24', 'id_25', 'id_26', 'id_32']

  imputed_numerical_df_columns.remove("TransactionID")
  imputed_numerical_df_columns.remove("isFraud")
  imputed_numerical_df_columns.remove("TransactionDT")
  # print('imputed_numerical_df_columns',imputed_numerical_df_columns)
  # ['TransactionAmt', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D4', 'D10', 'D15', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321']

  # ================================================================================
  # with pd.option_context('display.max_rows',100000):
  #   print('imputed_numerical_df',imputed_numerical_df.skew())

  for one_column_name in imputed_numerical_df_columns:
    one_skew=imputed_numerical_df[[one_column_name]].skew()[0]
    # print('one_skew',one_skew)
    # one_skew -0.007009102593962393

    if one_skew>1:
      # Apply log transform to reduce the skewness of data
      # log = lambda x: np.log10(x + 1 - min(0, x.min()))

      add_by_1=imputed_numerical_df[one_column_name]+1
      # print('add_by_1',add_by_1)

      min_in_column=imputed_numerical_df[one_column_name].min()
      # print('min_in_column',min_in_column)
      # min_in_column 0.424

      transformed=np.log10(add_by_1-min(0,min_in_column))
      imputed_numerical_df[one_column_name]=transformed

  # with pd.option_context('display.max_rows',100000):
  #   print('imputed_numerical_df',imputed_numerical_df.skew())

  return imputed_numerical_df

def see_feature_importance(trained_classifier,used_features):
  scores=trained_classifier.booster_.feature_importance(importance_type='gain')
  # print('scores',scores)
  # print('used_features',used_features)
 
  d={'scores':scores,
     'used_features':used_features}
  df=pd.DataFrame(d)

  with pd.option_context('display.max_rows',100000):
    print(df.sort_values(by=['scores'],axis=0,ascending=False))

def evaluate_by_deeplearning(train_d,val_d,model_save=False):
  # print("normalized_train_X",normalized_train_X)
  # print("smote_train_y",smote_train_y)
  
  net=Net(n_feature=number_of_features,n_hidden=100,n_output=number_of_labels).cuda()     # define the network

  pytorch_total_params=sum(p.numel() for p in net.parameters())
  pytorch_total_trainable_params=sum(p.numel() for p in net.parameters() if p.requires_grad)

  # optimizer=torch.optim.SGD(net.parameters(),lr=0.2)
  optimizer=torch.optim.SGD(net.parameters(),lr=0.1)
  # loss_func=FocalLoss(class_num=15)
  loss_func=torch.nn.MSELoss()  # this is for regression mean squared loss

  # ================================================================================
  torch_dataset=IEEE_Dataset(train_d)
  loader=data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
  )

  torch_test_dataset=IEEEVal_Dataset(val_d)
  test_loader=data.DataLoader(
    dataset=torch_test_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,           # mini batch size
    shuffle=True,                    # random shuffle for training
    num_workers=2,                   # subprocesses for loading data
  )

  # ================================================================================
  def train_by_batches():
    net.train()
    loss_list=[]
    for epoch in range(DEEP_LEARNING_EPOCH):   # train entire dataset 3 times
      for step,(batch_x,batch_y) in enumerate(loader):  # for each training step
        batch_x=batch_x.float().cuda()
        # print("batch_x",batch_x)
        # batch_x torch.Size([8000, 40])
        # batch_x=batch_x.view(BATCH_SIZE,number_of_features)
        batch_x=batch_x.view(batch_x.shape[0],number_of_features)
        # print("batch_x",batch_x.shape)
        # batch_x torch.Size([20, 1])
        batch_y=batch_y.float().cuda()
        # print("batch_y",batch_y)
        # batch_y torch.Size([5, 2])
      
        prediction=net(batch_x)     # input x and predict based on x
        # print("prediction",prediction.shape)
        # print("batch_y",batch_y.shape)
        # print("prediction",prediction)
        
        # prediction=prediction.view(5,4)

        loss=loss_func(prediction,batch_y)     # must be (1. nn output, 2. target)
        print("loss",loss.item())
        loss_list.append(loss.item())

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
      print('epoch',epoch)
        # print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',batch_x.numpy(), '| batch y: ', batch_y.numpy())
      # scheduler.step()
    torch.save({
      'model_state_dict': net.state_dict(),
      'optimizer_state_dict': optimizer.state_dict()},PATH)

    # model = TheModelClass(*args, **kwargs)
    # optimizer = TheOptimizerClass(*args, **kwargs)

    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    # model.eval()
    # model.train()
    
    plt.plot(loss_list)
    plt.show()
    # /mnt/external_disk/Capture_temp/2020_04_25_20:44:25.png

  # [120,80,....] -> [0.8,0.3,0.6] -> [1,0,1]

  # ================================================================================
  def test_by_batches():
    with torch.no_grad():

      checkpoint = torch.load("./trained_model.pth")
      net.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

      net.eval()

      all_targets=[]
      all_predictions=[]
      for step,(batch_x,batch_y) in enumerate(test_loader):  # for each training step
        batch_x=batch_x.float().cuda()
        # print("batch_x",batch_x.shape)
        # batch_x=batch_x.view(BATCH_SIZE,number_of_features)
        batch_x=batch_x.view(batch_x.shape[0],number_of_features)
        # print("batch_x",batch_x.shape)
        # batch_x torch.Size([20, 1])
        batch_y=batch_y.float().cuda()
        # print("batch_y",batch_y.shape)
        # batch_y torch.Size([5, 2])

        prediction=net(batch_x)     # input x and predict based on x
        # print("prediction",prediction.shape)
        # print("batch_y",batch_y.shape)
        # print("prediction",prediction)
        # print("batch_y",batch_y)
        
        one_int=np.random.randint(low=0,high=1321,size=1)
        pred_np=np.array(prediction.detach().cpu())
        # print("pred_np",pred_np.shape)
        pred_np=np.array(pred_np,dtype=np.float16)
        # print("pred_np",pred_np.tolist()[one_int[0]])
        # [[0.       0.1148   0.0265  ]
        #  [0.001524 0.3901   0.2874  ]
        #  [0.       0.1663   0.09485 ]
        #  ...
        #  [0.8755   0.5815   0.4675  ]
        #  [0.01794  0.4138   0.4548  ]
        #  [0.9707   0.645    0.2693  ]]
        target_np=np.array(batch_y.detach().cpu())
        target_np=np.array(target_np,dtype=np.float16)
        # print("target_np",target_np.tolist()[one_int[0]])

        pred_np=np.round_(pred_np,decimals=0,out=None)
        # pred_np=np.clip(pred_np,1,15,out=None)
        # print("target_np.shape[0]",target_np.shape[0])
        
        target_np_flat=target_np.reshape(-1)
        # print("target_np_flat",target_np_flat.shape)
        # print("target_np_flat",target_np_flat)
        
        pred_np_flat=pred_np.reshape(-1)
        # print("pred_np_flat",pred_np_flat.shape)
        # print("pred_np_flat",pred_np_flat)

        all_targets.extend(list(target_np_flat))
        all_predictions.extend(list(pred_np_flat))

        print((target_np_flat==pred_np_flat).sum())
        print((target_np_flat==pred_np_flat).sum()/target_np_flat.shape[0])
      print("all_targets",list(np.array(all_targets).reshape(-1)))
      print("all_predictions",list(np.array(all_predictions).reshape(-1)))

      # ================================================================================
      y_true=list(np.array(all_targets).reshape(-1))
      y_pred=list(np.array(all_predictions).reshape(-1))

      print('ROC accuracy: {}'.format(roc_auc_score(y_true,y_pred)))

      from sklearn.metrics import accuracy_score
      print("accuracy_score",accuracy_score(y_true, y_pred))

      from sklearn.metrics import confusion_matrix
      print("confusion_matrix",confusion_matrix(y_true, y_pred))

      sets=perf_measure(y_true,y_pred)
      print("perf_measure sets : (TP,FP,TN,FN)",sets)

      from sklearn.metrics import f1_score
      print("f1_score average='macro'",f1_score(y_true, y_pred, average='macro'))
      print("f1_score average='micro'",f1_score(y_true, y_pred, average='micro'))
      print("f1_score average='weighted'",f1_score(y_true, y_pred, average='weighted'))
      print("f1_score average='None'",f1_score(y_true, y_pred, average=None))
      print("f1_score zero_division=1",f1_score(y_true, y_pred, zero_division=1))

      from sklearn.metrics import recall_score
      print("recall_score average='macro'",recall_score(y_true, y_pred, average='macro'))
      print("recall_score average='micro'",recall_score(y_true, y_pred, average='micro'))
      print("recall_score average='weighted'",recall_score(y_true, y_pred, average='weighted'))
      print("recall_score average='None'",recall_score(y_true, y_pred, average=None))
      print("recall_score zero_division=1",recall_score(y_true, y_pred, zero_division=1))

      from sklearn.metrics import precision_score
      print("precision_score average='macro'",precision_score(y_true, y_pred, average='macro'))
      print("precision_score average='micro'",precision_score(y_true, y_pred, average='micro'))
      print("precision_score average='weighted'",precision_score(y_true, y_pred, average='weighted'))
      print("precision_score average='None'",precision_score(y_true, y_pred, average=None))
      print("precision_score zero_division=1",precision_score(y_true, y_pred, zero_division=1))

      # Creates a confusion matrix
      cm=confusion_matrix(y_true,y_pred)
      cm=cm/cm.astype(np.float).sum(axis=1)

      # Transform to df for easier plotting
      cm_df = pd.DataFrame(cm,index=['0','1'],columns=['0','1'])

      plt.figure(figsize=(5.5,4))
      sns.heatmap(cm_df, annot=True)
      plt.title('Linear Regression \nAccuracy:{0:.3f}'.format(accuracy_score(y_true,y_pred)))
      plt.ylabel('True label')
      plt.xlabel('Predicted label')
      plt.show()

  train_by_batches()
  test_by_batches()

def encode_CB(col1,col2,df1):
  # print('df1',df1)
  nm = col1+'_'+col2
  df1[nm] = (df1[col1].astype(str)+'_'+df1[col2].astype(str)).astype("category")
  return df1

def convert_time_delte(df):
  START_DATE=datetime.strptime('2017-11-30','%Y-%m-%d')
  df['DT_M']=df['TransactionDT'].apply(lambda x:(START_DATE+timedelta(seconds=x)))
  df['DT_M']=(df['DT_M'].dt.year-2017)*12+df['DT_M'].dt.month 
  return df

def categorical_other(imputed_categorical_df):
  
  low_threshold={
    "card1":100,
    "card2":50,
    "card3":20,
    "card5":20,
    "addr1":30,
    "id_31":10,
    "DeviceInfo":100}

  # ================================================================================
  # For train 

  column_collection=[]
  # print('imputed_categorical_df',imputed_categorical_df)
  for one_column_name in imputed_categorical_df:
    one_column_df=imputed_categorical_df[[one_column_name]]
    # print('one_column_df',one_column_df)
    
    # with pd.option_context('display.max_rows',100000):
    #   print('one_column_df^^\n',one_column_df.value_counts())
    #   print('')

    if one_column_name in ["card1","card2","card3","card5","addr1","id_31","DeviceInfo"]:

      # print("one_column_df.value_counts()\n",one_column_df.value_counts())
      # print('')

      # print('one_column_df.value_counts()<low_threshold[one_column_name]',one_column_df.value_counts()[one_column_df.value_counts()<low_threshold[one_column_name]])

      # one_column_df[one_column_df.value_counts()<low_threshold[one_column_name]]="Other"

      # df.apply(lambda x: x.mask(x.map(x.value_counts())<3, 'other') if x.name!='C' else x)

      # imputed_categorical_df=imputed_categorical_df.where(imputed_categorical_df.apply(lambda x: x.map(x.value_counts()))<low_threshold[one_column_name],"other")

      # aa=one_column_df.value_counts()[one_column_df.value_counts()<low_threshold[one_column_df]]
      # tt=one_column_df.value_counts()
      # print('tt',tt)
      
      for one_cate in list(map(lambda x:x[0],list(one_column_df.value_counts()[one_column_df.value_counts()<low_threshold[one_column_name]].to_frame().T.columns))):
        imputed_categorical_df[one_column_name]=imputed_categorical_df[one_column_name].replace(one_cate,'others')

      # imputed_categorical_df[one_column_name][one_column_df.value_counts()<low_threshold[one_column_name]]="others"
      # print('one_column_df.value_counts()<low_threshold[one_column_df]',tt)

      # column_collection.append(tt)
      # print('one_column_df.value_counts()<low_threshold[one_column_df]',list((tt[one_column_df.value_counts()<low_threshold[one_column_name]]).index))
      # afaf 

      # df['code'].cat.categories = ['one','two','three']
      # cat.add_categories('nullstr')


    else:
      continue
    
    # with pd.option_context('display.max_rows',100000):
    #   one_column_df=imputed_categorical_df[[one_column_name]]
    #   print('one_column_df^^\n',one_column_df.value_counts())
    #   print('')  

  return imputed_categorical_df

def full_train_by_lgbm(normalized_train_X,train_y,best_parameter_dict=None,model_save=False):

  X_train_=normalized_train_X
  y_train_=train_y

  # ================================================================================
  if best_parameter_dict!=None:

    params={
      'num_leaves':int(best_parameter_dict['num_leaves']),
      'n_estimators':int(best_parameter_dict['n_estimators']),
      'max_depth':int(best_parameter_dict['max_depth']),
      'subsample_for_bin':int(best_parameter_dict['subsample_for_bin']),
      'learning_rate':best_parameter_dict['learning_rate'],
      'subsample':best_parameter_dict['subsample'],
      'colsample_bytree':best_parameter_dict['colsample_bytree'],
      'min_child_samples':int(best_parameter_dict['min_child_samples']),
      'min_child_weight':best_parameter_dict['min_child_weight'],
      'min_split_gain':best_parameter_dict['min_split_gain'],
      'reg_lambda':best_parameter_dict['reg_lambda'],
      #'reg_alpha':best_parameter_dict['reg_alpha']}
    }

    # ================================================================================
    # create lgbm model with params

    lgbclf=lgb.LGBMClassifier(**params)

  else:
    params={
      'num_leaves':512,
      'n_estimators':512,
      'max_depth':9,
      'learning_rate':0.064,
      'subsample':0.85,
      'colsample_bytree':0.85,
      'boosting_type':'gbdt',
      'reg_alpha':0.3
    }

    # params={
    #   "num_leaves":512,
    #   "n_estimators":512,
    #   "max_depth":9,
    #   "learning_rate":0.064,
    #   "subsample":0.85,
    #   "colsample_bytree":0.85,
    #   "boosting_type":"gbdt",
    #   "reg_alpha":0.3,
    #   "reg_lamdba":0.243,
    #   "metric":"AUC"
    # }

    # params = {
    #   'num_leaves': 491,
    #   'min_child_weight': 0.03454472573214212,
    #   'feature_fraction': 0.3797454081646243,
    #   'bagging_fraction': 0.4181193142567742,
    #   'min_data_in_leaf': 106,
    #   'objective': 'binary',
    #   'max_depth': -1,
    #   'learning_rate': 0.006883242363721497,
    #   "boosting_type": "gbdt",
    #   "bagging_seed": 11,
    #   "metric": 'auc',
    #   "verbosity": -1,
    #   'reg_alpha': 0.3899927210061127,
    #   'reg_lambda': 0.6485237330340494,
    #   'random_state': 47
    # }

    lgbclf=lgb.LGBMClassifier(**params)

  # ================================================================================
  # Train lgb model with train dataset

  lgbclf.fit(X_train_,y_train_.values.ravel())
  
  # ================================================================================
  # Delete used data

  del X_train_,y_train_

  # ================================================================================
  if model_save==True:
    with open('./pickles/trained_model.pkl','wb') as f:
      pickle.dump(lgbclf,f)

  return lgbclf

def prediction_by_lgbm(normalized_train_X):
  # print('normalized_train_X',normalized_train_X.index)
  
  d={'TransactionID':normalized_train_X.index}
  TransactionID_df=pd.DataFrame(d)

  # ================================================================================
  # Load model
  
  with open('./pickles/trained_model.pkl','rb') as f:
    lgbclf=pickle.load(f)
  
  # ================================================================================
  # Make prediction on test dataset

  val=lgbclf.predict_proba(normalized_train_X)[:,1]
  # val=lgbclf.predict_proba(normalized_train_X)[:,:]
  # print("val",val)
  # print("val",val.shape)
  # val [0.00109754 0.01400123 0.00068512 ... 0.09750754 0.07325142 0.10856412]
  # val (506691,)

  # val_df=pd.DataFrame(val.round())
  val_df=pd.DataFrame(val)
  # print('val_df',val_df)
  #          0
  # 0     0.00
  # 1     0.00
  # 2     0.00
  # 3     0.00
  # 4     1.00
  # ...    ...
  # 13653 1.00
  # 13654 0.00
  # 13655 0.00
  # 13656 0.00
  # 13657 0.00

  # [13658 rows x 1 columns]

  prediction_df=pd.concat([TransactionID_df,val_df],axis=1)
  # print('prediction_df',prediction_df)
  #        TransactionID    0
  # 0            3663549 0.00
  # 1            3663550 0.00
  # 2            3663551 0.00
  # 3            3663552 0.00
  # 4            3663553 1.00
  # ...              ...  ...
  # 13653        3677202 1.00
  # 13654        3677203 0.00
  # 13655        3677204 0.00
  # 13656        3677205 0.00
  # 13657        3677206 0.00

  # [13658 rows x 2 columns]

  return prediction_df

def create_submission(predictions):
  sub=pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/sample_submission_original.csv')
  # print('sub',sub)
  # sub        TransactionID  isFraud
  # 0            3663549     0.50
  # 1            3663550     0.50
  # 2            3663551     0.50
  # 3            3663552     0.50
  # 4            3663553     0.50
  # ...              ...      ...
  # 13653        3677202     0.50
  # 13654        3677203     0.50
  # 13655        3677204     0.50
  # 13656        3677205     0.50
  # 13657        3677206     0.50

  # [13658 rows x 2 columns]

  sub=sub.sort_values(by=['TransactionID'],axis=0)

  merged=pd.merge(sub,predictions,on=['TransactionID'],how='left')
  del merged["isFraud"]
  merged=merged.rename(columns={0:'isFraud'})
  # print('merged',merged)
  #         TransactionID  isFraud
  # 0             3663549     0.00
  # 1             3663550     0.00
  # 2             3663551     0.00
  # 3             3663552     0.00
  # 4             3663553     1.00
  # ...               ...      ...
  # 506686        4170235      nan
  # 506687        4170236      nan
  # 506688        4170237      nan
  # 506689        4170238      nan
  # 506690        4170239      nan

  # [506691 rows x 2 columns]

  fn='./results_csv/sample_submission_'+str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))+'.csv'
  merged.to_csv(fn,sep=',',encoding='utf-8',index=False)

def new_feature_label_encoded_year_month_day(numerical_df):
  # print('numerical_df',numerical_df)

  START_DATE=datetime.strptime('2017-11-30','%Y-%m-%d')
  Ymd_temp_series=numerical_df['TransactionDT'].apply(lambda x:(START_DATE+timedelta(seconds=x)).strftime('%Y%m%d'))
  print('Ymd_temp_series',Ymd_temp_series)

  le=preprocessing.LabelEncoder()
  le.fit(Ymd_temp_series)
  LabelEncoder()
  list(le.classes_)
  ['amsterdam', 'paris', 'tokyo']
  le.transform(['tokyo', 'tokyo', 'paris'])

  df['DT_M']=(df['DT_M'].dt.year-2017)*12+df['DT_M'].dt.month 
  return df

def inspect_time_range(numerical_df):
  # print('numerical_df',numerical_df)

  # START_DATE=datetime.strptime('2017-11-30','%Y-%m-%d')
  # Ymd_temp_series=numerical_df['TransactionDT'].apply(lambda x:(START_DATE+timedelta(seconds=x)).strftime('%Y'))
  # print('Ymd_temp_series',Ymd_temp_series.value_counts())
  # 2018    453219
  # 2017    137321

  START_DATE=datetime.strptime('2017-01-01','%Y-%m-%d')
  Ymd_temp_series=numerical_df['TransactionDT'].apply(lambda x:(START_DATE+timedelta(seconds=x)).strftime('%Y'))
  # print('Ymd_temp_series',Ymd_temp_series.value_counts())
  # Ymd_temp_series 2017    590540

def test_to_datetime_with_unit_s_argument(numerical_df):

  print(pd.to_datetime(numerical_df['TransactionDT'],unit='s'))
  # TransactionID
  # 2987031.00   1970-01-02 00:09:58
  # 2987111.00   1970-01-02 00:33:03
  # 2987261.00   1970-01-02 01:08:12

def investigate_TransactionID_frequency_and_isFraud_frequency(csv_df):
  # print('csv_df',csv_df)
  #        TransactionID  TransactionDT  TransactionAmt  dist1  dist2     C1  \
  # 13285     2987031.00       86998.00          363.89  13.00    nan   1.00   
  # 483       2987111.00       88383.00           18.19    nan 109.00   1.00   
  print('list(csv_df["TransactionID"]))',len(list(csv_df["TransactionID"])))
  print(Counter(list(csv_df["TransactionID"])))

def investigate_correlation_in_features(numerical_train_df):
  corr=skewness_managed_numerical_train_df.corr()

  c1 = corr.abs().unstack()
  res=c1.sort_values(ascending = False).reset_index()
  res2=c1.sort_values(ascending = True).reset_index()
  # with pd.option_context('display.max_rows',100000):
  #   print('res',res.reset_index())
  #                  level_0         level_1    0
  #   0                DT_M            DT_M 1.00
  #   1                V116            V116 1.00
  #   2                V106            V106 1.00
  # with pd.option_context('display.max_rows',100000):
  #   print('res2',res2)

  duplicated_list=[]
  for i in range(res.shape[0]):
    # print(res.iloc[i,:])
    # level_0    DT_M
    # level_1    DT_M
    # 0          1.00
    
    # ================================================================================
    first_column_name=res.iloc[i,:]["level_0"]
    second_column_name=res.iloc[i,:]["level_1"]
    # print("first_column_name",first_column_name)
    # print("second_column_name",second_column_name)
    # first_column_name DT_M
    # second_column_name DT_M
    
    # ================================================================================
    # Sort for consistent order

    consistent_order_list=[first_column_name,second_column_name]
    consistent_order_list.sort()
    # print('consistent_order_list',consistent_order_list)

    # ================================================================================
    filtered_row=res[(res['level_0']==consistent_order_list[0])&(res['level_1']==consistent_order_list[1])]
    
    duplicated_list.append(filtered_row)

  concat_duplicated_list=pd.concat(duplicated_list)
  # print('concat_duplicated_list',concat_duplicated_list)

  # ================================================================================
  # Pickle
  # "Correlation in features" ~ "Correlation in features"

  with open('./pickles/concat_duplicated_list.pkl','wb') as f:
    pickle.dump(concat_duplicated_list,f)
  with open('./pickles/concat_duplicated_list.pkl','rb') as f:
    concat_duplicated_list=pickle.load(f)

  concat_duplicated_list=concat_duplicated_list.drop_duplicates(keep='first')
  concat_duplicated_list=concat_duplicated_list[concat_duplicated_list['level_0']!=concat_duplicated_list['level_1']]

  fn='./results_csv/numerical_data_features_correlation.csv'
  concat_duplicated_list.to_csv(fn,sep=',',encoding='utf-8',index=False)

  # with pd.option_context('display.max_rows',100000):
  #   print('concat_duplicated_list',concat_duplicated_list)

def visualize_correlation_in_features(numerical_train_df):
  corr=numerical_train_df.corr()
  fig,ax=plt.subplots(figsize=(15,15))
  aa=ax.matshow(corr,cmap=plt.get_cmap('Reds'))
  fig.colorbar(aa,ax=ax)
  plt.xticks(range(len(corr.columns)), corr.columns,rotation=90,fontsize=3)
  plt.yticks(range(len(corr.columns)), corr.columns,fontsize=3)
  fig.savefig('correlation_features.png',dpi=800)
  plt.close(fig)    # close the figure window
  # plt.show()

def find_columns_to_be_removed_which_has_high_correlation():

  # ================================================================================
  # Very strong correlation features

  nan_percent_df=pd.read_csv('/mnt/external_disk/Companies/side_project/Kaggle/00001_ieee-fraud-detection/my_code/src/results_csv/nan_percent.csv',encoding='utf-8')
  pre_sale=pd.read_csv('/mnt/external_disk/Companies/side_project/Kaggle/00001_ieee-fraud-detection/my_code/src/results_csv/numerical_data_features_correlation.csv',encoding='utf-8')
  # print('pre_sale',pre_sale)
  #       level_0 level_1    0
  # 0        V101     V95 1.00
  # 1        V279    V293 1.00
  # 2         C12      C7 1.00

  strong_correlation_pairs_df=pre_sale[pre_sale.iloc[:,2]>=0.8]


  to_be_dropped_columns=[]
  for i in range(strong_correlation_pairs_df.shape[0]):
    first_column_name=strong_correlation_pairs_df.iloc[i,0]
    second_column_name=strong_correlation_pairs_df.iloc[i,1]
    # print("first_column_name",first_column_name)
    # print("second_column_name",second_column_name)
    # first_column_name V101
    # second_column_name V95

    try:
      first_column_name_nan_percent=list(nan_percent_df[nan_percent_df["column_name"]==first_column_name]["nan_percent"])[0]
    except:
      # print('first_column_name',first_column_name)
      # print(traceback.format_exc())
      first_column_name_nan_percent=0.0
    
    try:
      second_column_name_nan_percent=list(nan_percent_df[nan_percent_df["column_name"]==second_column_name]["nan_percent"])[0]
    except:
      # print('second_column_name',second_column_name)
      # print(traceback.format_exc())
      second_column_name_nan_percent=0.0
    
    # print("first_column_name_nan_percent",first_column_name_nan_percent)
    # print("second_column_name_nan_percent",second_column_name_nan_percent)
    # first_column_name_nan_percent [0.05317167]
    # second_column_name_nan_percent [0.05317167]

    try:
      if first_column_name_nan_percent<=second_column_name_nan_percent:
        drop_column=second_column_name
      else:
        drop_column=first_column_name
    except:
      print(traceback.format_exc())
      continue
    
    to_be_dropped_columns.append(drop_column)
    
  to_be_dropped_columns=list(OrderedDict.fromkeys(to_be_dropped_columns))
  # print('to_be_dropped_columns',to_be_dropped_columns)
  # ['V95', 'V293', 'C7', 'V101', 'V97', 'V295', 'V132', 'V96', 'C8', 'C11', 'V134', 'V133', 'C2', 'V316', 'V318', 'V126', 'V317', 'V103', 'V18', 'C6', 'V296', 'V128', 'V16', 'V127', 'V32', 'C12', 'V308', 'V102', 'V306', 'V105', 'V106', 'C4', 'V28', 'V307', 'V11', 'V49', 'V292', 'V22', 'V72', 'V93', 'V58', 'V304', 'C14', 'V70', 'V91', 'V34', 'V94', 'V33', 'C10', 'V30', 'V299', 'V74', 'V81', 'V60', 'V52', 'V43', 'V21', 'V294', 'V64', 'V40', 'V71', 'V73', 'V321', 'V85', 'V63', 'V137', 'V57', 'V36', 'V280', 'V51', 'V79', 'V301', 'V13', 'C9', 'V113', 'V298', 'V129', 'V54', 'V76', 'V315', 'V104', 'V50', 'V90', 'V31', 'V5', 'V92', 'V69', 'V42', 'V20', 'V45', 'V84', 'V62', 'V289', 'V311', 'V83', 'V303', 'V131', 'V119', 'V38', 'V39', 'V100', 'V99', 'V48', 'V136', 'V80', 'V87', 'V59', 'V130', 'V26', 'V9', 'V287', 'V67', 'V24', 'V125', 'V110', 'D2', 'V17', 'V116']

def discard_high_correlation_columns(csv_train,train_columns_to_be_dropped):
  # print("csv_train",csv_train)
  # print("train_columns_to_be_dropped",train_columns_to_be_dropped)
  
  for one_column_name in list(csv_train.columns):
    if one_column_name in train_columns_to_be_dropped:
      del csv_train[one_column_name]

  # print("csv_train",csv_train)
  
  return csv_train

# ================================================================================
find_columns_to_be_removed_which_has_high_correlation()

# ================================================================================
# Load csv file

csv_train=load_csv_files_train()
csv_test=load_csv_files_test()
# print("csv_train",csv_train)
# print("csv_test",csv_test)

# ================================================================================
# 558investigate_TransactionID_frequency_and_isFraud_frequency(csv_train)

# ================================================================================
# check NaN in column and row

train_nan_ratio_df,train_columns_to_be_dropped=check_nan(csv_train,NAN_CRITERION,create_image=CREATE_IMAGE_ON_NAN_RATIO)
test_columns_to_be_dropped=list(map(lambda x:x.replace("id_","id-"),train_columns_to_be_dropped))
# print('nan_ratio_df',nan_ratio_df)

# ================================================================================
# Discard columns whose NaNs are too abundant

csv_train=discard_nan_columns(csv_train,train_columns_to_be_dropped)
csv_test=discard_nan_columns(csv_test,test_columns_to_be_dropped)
# print("csv_train",csv_train)
# print("csv_test",csv_test)

# ================================================================================
# Discard columns (high correlation)

train_columns_to_be_dropped=['V95', 'V293', 'C7', 'V101', 'V97', 'V295', 'V132', 'V96', 'C8', 'C11', 'V134', 'V133', 'C2', 'V316', 'V318', 'V126', 'V317', 'V103', 'V18', 'C6', 'V296', 'V128', 'V16', 'V127', 'V32', 'C12', 'V308', 'V102', 'V306', 'V105', 'V106', 'C4', 'V28', 'V307', 'V11', 'V49', 'V292', 'V22', 'V72', 'V93', 'V58', 'V304', 'C14', 'V70', 'V91', 'V34', 'V94', 'V33', 'C10', 'V30', 'V299', 'V74', 'V81', 'V60', 'V52', 'V43', 'V21', 'V294', 'V64', 'V40', 'V71', 'V73', 'V321', 'V85', 'V63', 'V137', 'V57', 'V36', 'V280', 'V51', 'V79', 'V301', 'V13', 'C9', 'V113', 'V298', 'V129', 'V54', 'V76', 'V315', 'V104', 'V50', 'V90', 'V31', 'V5', 'V92', 'V69', 'V42', 'V20', 'V45', 'V84', 'V62', 'V289', 'V311', 'V83', 'V303', 'V131', 'V119', 'V38', 'V39', 'V100', 'V99', 'V48', 'V136', 'V80', 'V87', 'V59', 'V130', 'V26', 'V9', 'V287', 'V67', 'V24', 'V125', 'V110', 'D2', 'V17', 'V116']

csv_train=discard_high_correlation_columns(csv_train,train_columns_to_be_dropped)
csv_test=discard_high_correlation_columns(csv_test,train_columns_to_be_dropped)

# # ================================================================================
# # Load pickle 
# # "Load csv file" ~ "Discard columns whose NaNs are too abundant"

# # with open('./pickles/csv_train_small.pkl','wb') as f:
# #   pickle.dump(csv_train,f)
# with open('./pickles/csv_train_small.pkl','rb') as f:
#   csv_train=pickle.load(f)

# # with open('./pickles/csv_test_small.pkl','wb') as f:
# #   pickle.dump(csv_test,f)
# with open('./pickles/csv_test_small.pkl','rb') as f:
#   csv_test=pickle.load(f)

# # with open('./pickles/csv_train_full.pkl','wb') as f:
# #   pickle.dump(csv_train,f)
# # with open('./pickles/csv_train_full.pkl','rb') as f:
# #   csv_train=pickle.load(f)

# # with open('./pickles/csv_test_full.pkl','wb') as f:
# #   pickle.dump(csv_test,f)
# # with open('./pickles/csv_test_full.pkl','rb') as f:
# #   csv_test=pickle.load(f)

# ================================================================================
# Separate full column data into categorical data and numerical data 

numerical_train_df,categorical_train_df=separate_full_column_data_into_categorical_and_numerical(csv_train)
numerical_test_df,categorical_test_df=separate_full_column_data_into_categorical_and_numerical(csv_test)
numerical_train_df=numerical_train_df.astype("float32")
numerical_test_df=numerical_test_df.astype("float32")
# print("numerical_train_df",numerical_train_df)
# print("categorical_train_df",categorical_train_df)
# print("numerical_test_df",numerical_test_df)
# print("categorical_test_df",categorical_test_df)

del csv_train
del csv_test
gc.collect()

# ================================================================================
# Inspect time range

# numerical_train_df=inspect_time_range(numerical_train_df)

# ================================================================================
# Inspect time range

# numerical_train_df=test_to_datetime_with_unit_s_argument(numerical_train_df)

# ================================================================================
# Convert time delta

numerical_train_df=convert_time_delte(numerical_train_df)
numerical_test_df=convert_time_delte(numerical_test_df)
# print("numerical_train_df",numerical_train_df)
# print("numerical_test_df",numerical_test_df)

# ================================================================================
# Insert new feature (year+month+day in label encoding form)

# numerical_train_df=new_feature_label_encoded_year_month_day(numerical_train_df)
# new_feature_label_encoded_year_month_day

# ================================================================================
# Impute null of categorical data by mode

imputed_categorical_train_df=impute_categorical_data_by_mode(categorical_train_df)
imputed_categorical_test_df=impute_categorical_data_by_mode(categorical_test_df)
# print('imputed_categorical_train_df',imputed_categorical_train_df)
# print('imputed_categorical_test_df',imputed_categorical_test_df)

del categorical_train_df
del categorical_test_df
gc.collect()

# ================================================================================
imputed_categorical_train_df=categorical_other(imputed_categorical_train_df)
imputed_categorical_test_df=categorical_other(imputed_categorical_test_df)
# print("imputed_categorical_train_df",imputed_categorical_train_df)
# print("imputed_categorical_test_df",imputed_categorical_test_df)

# ================================================================================
# Combine feature to create new features

first_new_feature_added=encode_CB('addr1','addr2',imputed_categorical_train_df)
imputed_categorical_train_df=encode_CB('addr1_addr2','P_emaildomain',first_new_feature_added)

first_new_feature_added=encode_CB('addr1','addr2',imputed_categorical_test_df)
imputed_categorical_test_df=encode_CB('addr1_addr2','P_emaildomain',first_new_feature_added)

# ================================================================================
# Impute null of numerical data by MICE

if IMPUTED_NUMERICAL_DATA_SOURCE=="function":
  if METHOD_FOR_IMPUTE_NUMERICAL_DATA=="mice":
    imputed_numerical_train_df=impute_numerical_data_by_MICE(numerical_train_df)
    imputed_numerical_test_df=impute_numerical_data_by_MICE(numerical_test_df)
  elif METHOD_FOR_IMPUTE_NUMERICAL_DATA=="mean":
    imputed_numerical_train_df=impute_numerical_data_by_mean(numerical_train_df,"train")
    imputed_numerical_test_df=impute_numerical_data_by_mean(numerical_test_df,"test")
elif IMPUTED_NUMERICAL_DATA_SOURCE=="pickle":
  with open('./pickles/imputed_numerical_train_df.pkl','rb') as f:
    imputed_numerical_train_df=pickle.load(f)
  with open('./pickles/imputed_numerical_test_df.pkl','rb') as f:
    imputed_numerical_test_df=pickle.load(f)

del numerical_train_df
del numerical_test_df
gc.collect()

# ================================================================================
# Manage skewness

# skewness_managed_numerical_df=manage_skewness(imputed_numerical_df)
# del imputed_numerical_df
# gc.collect()

skewness_managed_numerical_train_df=imputed_numerical_train_df
skewness_managed_numerical_test_df=imputed_numerical_test_df

del imputed_numerical_train_df
del imputed_numerical_test_df
gc.collect()

# ================================================================================
# pickle
# "Separate full column data into categorical data and numerical data" ~ "Manage skewness"


# del csv_train
# del csv_test
# gc.collect()

# # with open('./pickles/skewness_managed_numerical_train_df.pkl','wb') as f:
# #   pickle.dump(skewness_managed_numerical_train_df,f)
# with open('./pickles/skewness_managed_numerical_train_df.pkl','rb') as f:
#   skewness_managed_numerical_train_df=pickle.load(f)

# # with open('./pickles/imputed_categorical_train_df.pkl','wb') as f:
# #   pickle.dump(imputed_categorical_train_df,f)
# with open('./pickles/imputed_categorical_train_df.pkl','rb') as f:
#   imputed_categorical_train_df=pickle.load(f)

# # ================================================================================
# # with open('./pickles/skewness_managed_numerical_test_df.pkl','wb') as f:
# #   pickle.dump(skewness_managed_numerical_test_df,f)
# with open('./pickles/skewness_managed_numerical_test_df.pkl','rb') as f:
#   skewness_managed_numerical_test_df=pickle.load(f)

# # with open('./pickles/imputed_categorical_test_df.pkl','wb') as f:
# #   pickle.dump(imputed_categorical_test_df,f)
# with open('./pickles/imputed_categorical_test_df.pkl','rb') as f:
#   imputed_categorical_test_df=pickle.load(f)

# ================================================================================
# Correlation in features

# visualize_correlation_in_features(skewness_managed_numerical_train_df)

# investigate_correlation_in_features(skewness_managed_numerical_train_df)

# ================================================================================
# Compress data

# csv_train_identity=reduce_mem_usage(csv_train_identity)
# compressed_imputed_numerical_df=reduce_mem_usage(imputed_numerical_df)

# ================================================================================
# Encode categorical data using LabelEncoder

if ENCODED_CATEGORICAL_DATA_SOURCE=="function":
  encoded_categorical_train_data=encode_categorical_data_using_LabelEncoder(imputed_categorical_train_df,"train")
  encoded_categorical_test_data=encode_categorical_data_using_LabelEncoder(imputed_categorical_test_df,"test")
elif ENCODED_CATEGORICAL_DATA_SOURCE=="pickle":
  with open('./pickles/imputed_categorical_train_df.pkl','rb') as f:
    encoded_categorical_train_data=pickle.load(f)
  with open('./pickles/imputed_categorical_test_df.pkl','rb') as f:
    encoded_categorical_test_data=pickle.load(f)

del imputed_categorical_train_df
del imputed_categorical_test_df
gc.collect()

# ================================================================================
# Concatenate numerical and categorical data

concated_numerical_categorical_train_data=concat_numerical_and_categorical_data(skewness_managed_numerical_train_df.astype("float32"),encoded_categorical_train_data)
concated_numerical_categorical_test_data=concat_numerical_and_categorical_data(skewness_managed_numerical_test_df.astype("float32"),encoded_categorical_test_data)
# print('concated_numerical_categorical_data',concated_numerical_categorical_data)

del skewness_managed_numerical_train_df
del encoded_categorical_train_data
del skewness_managed_numerical_test_df
del encoded_categorical_test_data
gc.collect()

# ================================================================================
# Perform normalization on all columns

# print('smote_train_X',smote_train_X)
# normalized_train_X,train_y=perform_normalization(concated_numerical_categorical_data)

# del concated_numerical_categorical_data
# gc.collect()

# Prepare normalized_train_X, train_y
train_y=concated_numerical_categorical_train_data[["isFraud"]]
del concated_numerical_categorical_train_data["isFraud"]

normalized_train_X=concated_numerical_categorical_train_data
del concated_numerical_categorical_train_data
gc.collect()

# Prepare normalized_test_X
normalized_test_X=concated_numerical_categorical_test_data
del concated_numerical_categorical_test_data
gc.collect()

# ================================================================================
# Split train and validation before oversampling

# full_train_X,full_train_y,full_validation_X,full_validation_y=split_train_and_validation_before_oversampling(concated_numerical_categorical_data)
# del concated_numerical_categorical_data
# gc.collect()

# ================================================================================
# Perform resampling

if RESAMPLING_USE==True:
  if RESAMPLING=="oversampling_smote":
    smote_train_X,smote_train_y=perform_oversampling(normalized_train_X,train_y)
  elif RESAMPLING=="oversampling_resampling":
    smote_train_X,smote_train_y=perform_oversampling_resampling(normalized_train_X,train_y)
  elif RESAMPLING=="undersampling_resampling":
    smote_train_X,smote_train_y=perform_undersampling_resampling(normalized_train_X,train_y)
else:
  smote_train_X,smote_train_y=normalized_train_X,train_y
  smote_test_X=normalized_test_X

del normalized_train_X
del train_y
del normalized_test_X
gc.collect()

# # ================================================================================
# # Test evaluation (Comment/Uncomment)

# trained_classifier=evaluate_by_lgbm(normalized_train_X,smote_train_y,full_validation_X,full_validation_y)

# ================================================================================
# Hyperparameter tuning (Comment/Uncomment)

if HYPERPARAMETER_TUNIING_LGBM==True:
  optimize_hyperparameter_of_lgbm(smote_train_X,smote_train_y)
  # {'colsample_bytree': 0.8041909572677464, 'learning_rate': 0.0030994569394493655, 'max_depth': 3.0, 'min_child_samples': 238.0, 'min_child_weight': 0.6124238031349085, 'min_split_gain': 0.7073711101457963, 'n_estimators': 3800.0, 'num_leaves': 189.0, 'reg_lambda': 9.564956408961201, 'subsample': 0.9465343997925819, 'subsample_for_bin': 8034.1155781467105}

# ================================================================================
# Validation (best parameter, oversampling)

if USE_VALIDATION==True:
  if HYPERPARAMETER_TUNIING_LGBM_USE==True:
    with open('./pickles/best_parameter_dict.pkl','rb') as f:
      best_parameter_dict=pickle.load(f)
    trained_classifier=evaluate_by_lgbm(smote_train_X,smote_train_y,best_parameter_dict,model_save=True)
  elif HYPERPARAMETER_TUNIING_LGBM_USE==False:
    trained_classifier=evaluate_by_lgbm(smote_train_X,smote_train_y,model_save=True)
  
  # ================================================================================
  # See feature importance

  # see_feature_importance(trained_classifier,list(normalized_train_X.columns))

# ================================================================================
# Train on full data

if USE_TRAIN==True:
  if HYPERPARAMETER_TUNIING_LGBM_USE==True:
    with open('./pickles/best_parameter_dict.pkl','rb') as f:
      best_parameter_dict=pickle.load(f)
    trained_classifier=full_train_by_lgbm(smote_train_X,smote_train_y,best_parameter_dict,model_save=True)
  elif HYPERPARAMETER_TUNIING_LGBM_USE==False:
    trained_classifier=full_train_by_lgbm(smote_train_X,smote_train_y,model_save=True)

  # ================================================================================
  # See feature importance

  # see_feature_importance(trained_classifier,list(normalized_train_X.columns))

# ================================================================================
# Make prediction 

if USE_TEST==True:
  predictions=prediction_by_lgbm(smote_test_X)

# ================================================================================
# Make submission 

if USE_TEST==True:
  create_submission(predictions)

# ================================================================================
# print("normalized_train_X",normalized_train_X)
# print("smote_train_y",smote_train_y)
# print("full_validation_X",full_validation_X)
# print("full_validation_y",full_validation_y)

# train_d=pd.concat([normalized_train_X,smote_train_y],axis=1)
# val_d=pd.concat([full_validation_X,full_validation_y],axis=1)

# del normalized_train_X,smote_train_y
# gc.collect()
# del full_validation_X,full_validation_y
# gc.collect()

# trained_classifier=evaluate_by_deeplearning(train_d,val_d,model_save=True)

