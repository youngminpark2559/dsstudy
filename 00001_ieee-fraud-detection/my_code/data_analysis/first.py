# conda activate py36_django_bare && \
# cd /mnt/external_disk/Companies/side_project/Kaggle/00001_ieee-fraud-detection/my_code/data_analysis && \
# rm e.l && python first.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import pickle
from sklearn.metrics import roc_auc_score
import optuna
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from scipy.stats import rankdata,ks_2samp
from sklearn import preprocessing
import numpy as np
from collections import Counter
from datetime import datetime,timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
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

def merge_transaction_identity(csv_train_identity,csv_train_transaction_original):
  merged=pd.merge(csv_train_transaction_original,csv_train_identity,on=['TransactionID'],how='left')
  # print('merged',merged)
  #         TransactionID  isFraud  TransactionDT  TransactionAmt ProductCD  \
  # 0             2987000        0          86400           68.50         W   
  # 1             2987001        0          86401           29.00         W   
  # 2             2987002        0          86469           59.00         W   
  # 3             2987003        0          86499           50.00         W   
  # 4             2987004        0          86506           50.00         H   
  # ...               ...      ...            ...             ...       ...   
  # 590535        3577535        0       15811047           49.00         W   
  # 590536        3577536        0       15811049           39.50         W   
  # 590537        3577537        0       15811079           30.95         W   
  # 590538        3577538        0       15811088          117.00         W   
  # 590539        3577539        0       15811131          279.95         W   

  #         card1  card2  card3       card4  card5   card6  addr1  addr2  dist1  \
  # 0       13926    NaN  150.0    discover  142.0  credit  315.0   87.0   19.0   
  # 1        2755  404.0  150.0  mastercard  102.0  credit  325.0   87.0    NaN   
  # 2        4663  490.0  150.0        visa  166.0   debit  330.0   87.0  287.0   
  # 3       18132  567.0  150.0  mastercard  117.0   debit  476.0   87.0    NaN   
  # 4        4497  514.0  150.0  mastercard  102.0  credit  420.0   87.0    NaN   
  # ...       ...    ...    ...         ...    ...     ...    ...    ...    ...   
  # 590535   6550    NaN  150.0        visa  226.0   debit  272.0   87.0   48.0   
  # 590536  10444  225.0  150.0  mastercard  224.0   debit  204.0   87.0    NaN   
  # 590537  12037  595.0  150.0  mastercard  224.0   debit  231.0   87.0    NaN   
  # 590538   7826  481.0  150.0  mastercard  224.0   debit  387.0   87.0    3.0   
  # 590539  15066  170.0  150.0  mastercard  102.0  credit  299.0   87.0    NaN   

  #         dist2 P_emaildomain R_emaildomain   C1   C2   C3   C4   C5   C6   C7  \
  # 0         NaN           NaN           NaN  1.0  1.0  0.0  0.0  0.0  1.0  0.0   
  # 1         NaN     gmail.com           NaN  1.0  1.0  0.0  0.0  0.0  1.0  0.0   
  # 2         NaN   outlook.com           NaN  1.0  1.0  0.0  0.0  0.0  1.0  0.0   
  # 3         NaN     yahoo.com           NaN  2.0  5.0  0.0  0.0  0.0  4.0  0.0   
  # 4         NaN     gmail.com           NaN  1.0  1.0  0.0  0.0  0.0  1.0  0.0   
  # ...       ...           ...           ...  ...  ...  ...  ...  ...  ...  ...   
  # 590535    NaN           NaN           NaN  2.0  1.0  0.0  0.0  1.0  0.0  0.0   
  # 590536    NaN     gmail.com           NaN  1.0  1.0  0.0  0.0  0.0  1.0  0.0   
  # 590537    NaN     gmail.com           NaN  1.0  1.0  0.0  0.0  1.0  1.0  0.0   
  # 590538    NaN       aol.com           NaN  1.0  1.0  0.0  0.0  0.0  3.0  0.0   
  # 590539    NaN     gmail.com           NaN  2.0  1.0  0.0  0.0  1.0  1.0  0.0   

  #         C8   C9  C10  C11  C12   C13  C14     D1     D2    D3    D4   D5  D6  \
  # 0       0.0  1.0  0.0  2.0  0.0   1.0  1.0   14.0    NaN  13.0   NaN  NaN NaN   
  # 1       0.0  0.0  0.0  1.0  0.0   1.0  1.0    0.0    NaN   NaN   0.0  NaN NaN   
  # 2       0.0  1.0  0.0  1.0  0.0   1.0  1.0    0.0    NaN   NaN   0.0  NaN NaN   
  # 3       0.0  1.0  0.0  1.0  0.0  25.0  1.0  112.0  112.0   0.0  94.0  0.0 NaN   
  # 4       1.0  0.0  1.0  1.0  0.0   1.0  1.0    0.0    NaN   NaN   NaN  NaN NaN   
  # ...     ...  ...  ...  ...  ...   ...  ...    ...    ...   ...   ...  ...  ..   
  # 590535  0.0  2.0  0.0  1.0  0.0   3.0  2.0   29.0   29.0  30.0   NaN  NaN NaN   
  # 590536  0.0  1.0  0.0  1.0  0.0   1.0  1.0    0.0    NaN   NaN   0.0  NaN NaN   
  # 590537  0.0  1.0  0.0  1.0  0.0   1.0  1.0    0.0    NaN   NaN   0.0  NaN NaN   
  # 590538  0.0  2.0  0.0  1.0  1.0   5.0  1.0   22.0   22.0   0.0  22.0  0.0 NaN   
  # 590539  0.0  2.0  0.0  1.0  0.0   1.0  1.0    0.0    NaN   0.0   1.0  0.0 NaN   

  #         D7  D8  D9   D10    D11  D12  D13  D14    D15   M1   M2   M3   M4  \
  # 0      NaN NaN NaN  13.0   13.0  NaN  NaN  NaN    0.0    T    T    T   M2   
  # 1      NaN NaN NaN   0.0    NaN  NaN  NaN  NaN    0.0  NaN  NaN  NaN   M0   
  # 2      NaN NaN NaN   0.0  315.0  NaN  NaN  NaN  315.0    T    T    T   M0   
  # 3      NaN NaN NaN  84.0    NaN  NaN  NaN  NaN  111.0  NaN  NaN  NaN   M0   
  # 4      NaN NaN NaN   NaN    NaN  NaN  NaN  NaN    NaN  NaN  NaN  NaN  NaN   
  # ...     ..  ..  ..   ...    ...  ...  ...  ...    ...  ...  ...  ...  ...   
  # 590535 NaN NaN NaN  56.0   56.0  NaN  NaN  NaN   56.0    T    T    T   M0   
  # 590536 NaN NaN NaN   0.0    0.0  NaN  NaN  NaN    0.0    T    F    F   M0   
  # 590537 NaN NaN NaN   0.0    0.0  NaN  NaN  NaN    0.0    T    F    F  NaN   
  # 590538 NaN NaN NaN  22.0   22.0  NaN  NaN  NaN   22.0    T    T    T   M0   
  # 590539 NaN NaN NaN   1.0    0.0  NaN  NaN  NaN    1.0    T    F    F  NaN   

  #         M5   M6   M7   M8   M9   V1   V2   V3   V4   V5   V6   V7   V8   V9  \
  # 0         F    T  NaN  NaN  NaN  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0   
  # 1         T    T  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   
  # 2         F    F    F    F    F  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0   
  # 3         T    F  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   
  # 4       NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   
  # ...     ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...   
  # 590535    T    F    F    F    T  1.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  1.0   
  # 590536    F    T    F    F    F  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0   
  # 590537  NaN    T  NaN  NaN  NaN  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0   
  # 590538    F    T  NaN  NaN  NaN  1.0  1.0  1.0  2.0  2.0  1.0  1.0  1.0  1.0   
  # 590539  NaN    T    F    F    F  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0   

  #         V10  V11  V12  V13  V14  V15  V16  V17  V18  V19  V20  V21  V22  V23  \
  # 0       0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0   
  # 1       NaN  NaN  0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0   
  # 2       0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0   
  # 3       NaN  NaN  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0   
  # 4       NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   
  # ...     ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...   
  # 590535  0.0  0.0  2.0  2.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   
  # 590536  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0   
  # 590537  1.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0   
  # 590538  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  2.0  2.0  0.0  0.0  1.0   
  # 590539  1.0  1.0  2.0  2.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  2.0   

  #         V24  V25  V26  V27  V28  V29  V30  V31  V32  V33  V34  V35  V36  V37  \
  # 0       1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  NaN  NaN  NaN   
  # 1       1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   
  # 2       1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0   
  # 3       1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0   
  # 4       NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   
  # ...     ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...   
  # 590535  1.0  1.0  1.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  NaN  NaN  NaN   
  # 590536  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0   
  # 590537  1.0  1.0  1.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0   
  # 590538  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0   
  # 590539  2.0  1.0  1.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  2.0  2.0  1.0   

  #         V38  V39  V40  V41  V42  V43  V44  V45  V46  V47  V48  V49  V50  V51  \
  # 0       NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   
  # 1       1.0  0.0  0.0  1.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0   
  # 2       1.0  0.0  0.0  1.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0   
  # 3       1.0  0.0  0.0  1.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0   
  # 4       NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   
  # ...     ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...   
  # 590535  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   
  # 590536  1.0  0.0  0.0  1.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0   
  # 590537  1.0  0.0  0.0  1.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0   
  # 590538  1.0  0.0  0.0  1.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0   
  # 590539  1.0  0.0  0.0  1.0  0.0  0.0  2.0  2.0  1.0  1.0  1.0  1.0  0.0  0.0   

  #         V52  V53  V54  V55  V56  V57  V58  V59  V60  V61  V62  V63  V64  V65  \
  # 0       NaN  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0   
  # 1       0.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0   
  # 2       0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0   
  # 3       0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0   
  # 4       NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   
  # ...     ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...   
  # 590535  NaN  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   
  # 590536  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0   
  # 590537  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0   
  # 590538  1.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  2.0  2.0  0.0  0.0  1.0   
  # 590539  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0   

  #         V66  V67  V68  V69  V70  V71  V72  V73  V74  V75  V76  V77  V78  V79  \
  # 0       1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0   
  # 1       1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0   
  # 2       1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0   
  # 3       1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0   
  # 4       NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   
  # ...     ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...   
  # 590535  1.0  1.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  2.0  2.0  1.0  1.0  0.0   
  # 590536  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0   
  # 590537  1.0  1.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0   
  # 590538  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0  0.0   
  # 590539  1.0  1.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  2.0  2.0  1.0  1.0  0.0   

  #         V80  V81  V82  V83  V84  V85  V86  V87  V88  V89  V90  V91  V92  V93  \
  # 0       0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0   
  # 1       0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0   
  # 2       0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0   
  # 3       0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0   
  # 4       NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   
  # ...     ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...   
  # 590535  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  0.0  1.0  1.0  0.0  0.0   
  # 590536  0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0   
  # 590537  0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0  1.0  1.0  0.0  0.0   
  # 590538  0.0  0.0  2.0  2.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0   
  # 590539  0.0  0.0  1.0  1.0  0.0  0.0  2.0  2.0  1.0  0.0  1.0  1.0  0.0  0.0   

  #         V94  V95   V96   V97  V98   V99  V100  V101  V102  V103  V104  V105  \
  # 0       0.0  0.0   1.0   0.0  0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   
  # 1       0.0  0.0   0.0   0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # 2       0.0  0.0   0.0   0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # 3       0.0  1.0  48.0  28.0  0.0  10.0   4.0   1.0  38.0  24.0   0.0   0.0   
  # 4       NaN  0.0   0.0   0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # ...     ...  ...   ...   ...  ...   ...   ...   ...   ...   ...   ...   ...   
  # 590535  0.0  0.0   1.0   0.0  0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # 590536  0.0  0.0   0.0   0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # 590537  0.0  0.0   0.0   0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # 590538  0.0  1.0   4.0   1.0  1.0   1.0   1.0   0.0   3.0   0.0   0.0   0.0   
  # 590539  0.0  1.0   1.0   1.0  0.0   0.0   0.0   0.0   0.0   0.0   1.0   1.0   

  #         V106  V107  V108  V109  V110  V111  V112  V113  V114  V115  V116  \
  # 0        0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 1        0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 2        0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 3        0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 4        0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # ...      ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 590535   0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 590536   0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 590537   0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 590538   0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 590539   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   

  #         V117  V118  V119  V120  V121  V122  V123  V124  V125        V126  \
  # 0        1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0    0.000000   
  # 1        1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0    0.000000   
  # 2        1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0    0.000000   
  # 3        1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   50.000000   
  # 4        1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0    0.000000   
  # ...      ...   ...   ...   ...   ...   ...   ...   ...   ...         ...   
  # 590535   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0    0.000000   
  # 590536   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0    0.000000   
  # 590537   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0    0.000000   
  # 590538   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0  117.000000   
  # 590539   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0  279.950012   

  #               V127        V128   V129        V130   V131  V132    V133  \
  # 0        117.000000    0.000000    0.0    0.000000    0.0   0.0   117.0   
  # 1          0.000000    0.000000    0.0    0.000000    0.0   0.0     0.0   
  # 2          0.000000    0.000000    0.0    0.000000    0.0   0.0     0.0   
  # 3       1758.000000  925.000000    0.0  354.000000  135.0  50.0  1404.0   
  # 4          0.000000    0.000000    0.0    0.000000    0.0   0.0     0.0   
  # ...             ...         ...    ...         ...    ...   ...     ...   
  # 590535    47.950001    0.000000    0.0   47.950001    0.0   0.0     0.0   
  # 590536     0.000000    0.000000    0.0    0.000000    0.0   0.0     0.0   
  # 590537     0.000000    0.000000    0.0    0.000000    0.0   0.0     0.0   
  # 590538  1035.500000  117.000000  117.0  117.000000  117.0   0.0   918.5   
  # 590539   279.950012  279.950012    0.0    0.000000    0.0   0.0     0.0   

  #         V134        V135        V136        V137  V138  V139  V140  V141  \
  # 0         0.0    0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   
  # 1         0.0    0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   
  # 2         0.0    0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   
  # 3       790.0    0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   
  # 4         0.0    0.000000    0.000000    0.000000   0.0   0.0   0.0   0.0   
  # ...       ...         ...         ...         ...   ...   ...   ...   ...   
  # 590535    0.0    0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   
  # 590536    0.0    0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   
  # 590537    0.0    0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   
  # 590538    0.0    0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   
  # 590539    0.0  279.950012  279.950012  279.950012   NaN   NaN   NaN   NaN   

  #         V142  V143  V144   V145  V146  V147  V148  V149    V150  V151  V152  \
  # 0        NaN   NaN   NaN    NaN   NaN   NaN   NaN   NaN     NaN   NaN   NaN   
  # 1        NaN   NaN   NaN    NaN   NaN   NaN   NaN   NaN     NaN   NaN   NaN   
  # 2        NaN   NaN   NaN    NaN   NaN   NaN   NaN   NaN     NaN   NaN   NaN   
  # 3        NaN   NaN   NaN    NaN   NaN   NaN   NaN   NaN     NaN   NaN   NaN   
  # 4        0.0   6.0  18.0  140.0   0.0   0.0   0.0   0.0  1803.0  49.0  64.0   
  # ...      ...   ...   ...    ...   ...   ...   ...   ...     ...   ...   ...   
  # 590535   NaN   NaN   NaN    NaN   NaN   NaN   NaN   NaN     NaN   NaN   NaN   
  # 590536   NaN   NaN   NaN    NaN   NaN   NaN   NaN   NaN     NaN   NaN   NaN   
  # 590537   NaN   NaN   NaN    NaN   NaN   NaN   NaN   NaN     NaN   NaN   NaN   
  # 590538   NaN   NaN   NaN    NaN   NaN   NaN   NaN   NaN     NaN   NaN   NaN   
  # 590539   NaN   NaN   NaN    NaN   NaN   NaN   NaN   NaN     NaN   NaN   NaN   

  #         V153  V154  V155  V156  V157  V158          V159           V160  V161  \
  # 0        NaN   NaN   NaN   NaN   NaN   NaN           NaN            NaN   NaN   
  # 1        NaN   NaN   NaN   NaN   NaN   NaN           NaN            NaN   NaN   
  # 2        NaN   NaN   NaN   NaN   NaN   NaN           NaN            NaN   NaN   
  # 3        NaN   NaN   NaN   NaN   NaN   NaN           NaN            NaN   NaN   
  # 4        0.0   0.0   0.0   0.0   0.0   0.0  15557.990234  169690.796875   0.0   
  # ...      ...   ...   ...   ...   ...   ...           ...            ...   ...   
  # 590535   NaN   NaN   NaN   NaN   NaN   NaN           NaN            NaN   NaN   
  # 590536   NaN   NaN   NaN   NaN   NaN   NaN           NaN            NaN   NaN   
  # 590537   NaN   NaN   NaN   NaN   NaN   NaN           NaN            NaN   NaN   
  # 590538   NaN   NaN   NaN   NaN   NaN   NaN           NaN            NaN   NaN   
  # 590539   NaN   NaN   NaN   NaN   NaN   NaN           NaN            NaN   NaN   

  #         V162  V163   V164    V165    V166  V167  V168  V169  V170  V171  V172  \
  # 0        NaN   NaN    NaN     NaN     NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 1        NaN   NaN    NaN     NaN     NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 2        NaN   NaN    NaN     NaN     NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 3        NaN   NaN    NaN     NaN     NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 4        0.0   0.0  515.0  5155.0  2840.0   0.0   0.0   0.0   1.0   1.0   0.0   
  # ...      ...   ...    ...     ...     ...   ...   ...   ...   ...   ...   ...   
  # 590535   NaN   NaN    NaN     NaN     NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590536   NaN   NaN    NaN     NaN     NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590537   NaN   NaN    NaN     NaN     NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590538   NaN   NaN    NaN     NaN     NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590539   NaN   NaN    NaN     NaN     NaN   NaN   NaN   NaN   NaN   NaN   NaN   

  #         V173  V174  V175  V176  V177  V178  V179  V180  V181  V182  V183  \
  # 0        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 1        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 2        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 3        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 4        0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # ...      ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 590535   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590536   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590537   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590538   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590539   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   

  #         V184  V185  V186  V187  V188  V189  V190  V191  V192  V193  V194  \
  # 0        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 1        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 2        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 3        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 4        0.0   0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # ...      ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 590535   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590536   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590537   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590538   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590539   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   

  #         V195  V196  V197  V198  V199  V200  V201  V202  V203  V204  V205  \
  # 0        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 1        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 2        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 3        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 4        1.0   1.0   1.0   1.0   1.0   1.0   1.0   0.0   0.0   0.0   0.0   
  # ...      ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 590535   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590536   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590537   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590538   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590539   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   

  #         V206  V207  V208  V209  V210  V211  V212  V213  V214  V215  V216  \
  # 0        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 1        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 2        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 3        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 4        0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # ...      ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 590535   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590536   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590537   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590538   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590539   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   

  #         V217  V218  V219  V220  V221  V222  V223  V224  V225  V226  V227  \
  # 0        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 1        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 2        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 3        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 4        0.0   0.0   0.0   0.0   1.0   1.0   0.0   0.0   0.0   0.0   0.0   
  # ...      ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 590535   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590536   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590537   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590538   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590539   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   

  #         V228  V229  V230  V231  V232  V233  V234  V235  V236  V237  V238  \
  # 0        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 1        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 2        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 3        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 4        1.0   1.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # ...      ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 590535   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590536   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590537   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590538   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590539   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   

  #         V239  V240  V241  V242  V243  V244  V245  V246  V247  V248  V249  \
  # 0        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 1        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 2        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 3        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 4        0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # ...      ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 590535   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590536   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590537   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590538   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590539   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   

  #         V250  V251  V252  V253  V254  V255  V256  V257  V258  V259  V260  \
  # 0        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 1        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 2        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 3        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 4        1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # ...      ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 590535   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590536   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590537   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590538   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590539   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   

  #         V261  V262  V263  V264  V265  V266  V267  V268  V269  V270  V271  \
  # 0        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 1        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 2        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 3        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 4        1.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # ...      ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 590535   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590536   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590537   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590538   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590539   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   

  #         V272  V273  V274  V275  V276  V277  V278  V279  V280  V281  V282  \
  # 0        NaN   NaN   NaN   NaN   NaN   NaN   NaN   0.0   0.0   0.0   1.0   
  # 1        NaN   NaN   NaN   NaN   NaN   NaN   NaN   0.0   0.0   0.0   1.0   
  # 2        NaN   NaN   NaN   NaN   NaN   NaN   NaN   0.0   0.0   0.0   1.0   
  # 3        NaN   NaN   NaN   NaN   NaN   NaN   NaN   1.0  28.0   0.0   0.0   
  # 4        0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   
  # ...      ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 590535   NaN   NaN   NaN   NaN   NaN   NaN   NaN   0.0   0.0   0.0   1.0   
  # 590536   NaN   NaN   NaN   NaN   NaN   NaN   NaN   0.0   0.0   0.0   1.0   
  # 590537   NaN   NaN   NaN   NaN   NaN   NaN   NaN   0.0   0.0   0.0   1.0   
  # 590538   NaN   NaN   NaN   NaN   NaN   NaN   NaN   1.0   1.0   0.0   2.0   
  # 590539   NaN   NaN   NaN   NaN   NaN   NaN   NaN   1.0   1.0   0.0   2.0   

  #         V283  V284  V285  V286  V287  V288  V289  V290  V291  V292  V293  \
  # 0        1.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   1.0   1.0   0.0   
  # 1        1.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   1.0   1.0   0.0   
  # 2        1.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   1.0   1.0   0.0   
  # 3        0.0   0.0  10.0   0.0   4.0   0.0   0.0   1.0   1.0   1.0   1.0   
  # 4        1.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   1.0   1.0   0.0   
  # ...      ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 590535   1.0   0.0   1.0   0.0   0.0   1.0   1.0   1.0   1.0   1.0   0.0   
  # 590536   1.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   1.0   1.0   0.0   
  # 590537   1.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   1.0   1.0   0.0   
  # 590538   7.0   1.0   5.0   0.0   1.0   1.0   1.0   1.0   2.0   1.0   0.0   
  # 590539   2.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   1.0   1.0   0.0   

  #         V294  V295  V296  V297  V298  V299  V300  V301  V302  V303  V304  \
  # 0        1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # 1        0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # 2        0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # 3       38.0  24.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # 4        0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   1.0   1.0   
  # ...      ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 590535   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # 590536   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # 590537   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # 590538  11.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # 590539   0.0   0.0   1.0   1.0   1.0   1.0   1.0   1.0   0.0   0.0   0.0   

  #         V305        V306         V307        V308   V309        V310  V311  \
  # 0        1.0    0.000000   117.000000    0.000000    0.0    0.000000   0.0   
  # 1        1.0    0.000000     0.000000    0.000000    0.0    0.000000   0.0   
  # 2        1.0    0.000000     0.000000    0.000000    0.0    0.000000   0.0   
  # 3        1.0   50.000000  1758.000000  925.000000    0.0  354.000000   0.0   
  # 4        1.0    0.000000     0.000000    0.000000    0.0    0.000000   0.0   
  # ...      ...         ...          ...         ...    ...         ...   ...   
  # 590535   1.0    0.000000    47.950001    0.000000    0.0   47.950001   0.0   
  # 590536   1.0    0.000000     0.000000    0.000000    0.0    0.000000   0.0   
  # 590537   1.0    0.000000     0.000000    0.000000    0.0    0.000000   0.0   
  # 590538   1.0  117.000000  2903.500000  117.000000  117.0  669.500000   0.0   
  # 590539   1.0  279.950012   279.950012  279.950012    0.0    0.000000   0.0   

  #         V312        V313        V314        V315  V316    V317   V318  \
  # 0         0.0    0.000000    0.000000    0.000000   0.0   117.0    0.0   
  # 1         0.0    0.000000    0.000000    0.000000   0.0     0.0    0.0   
  # 2         0.0    0.000000    0.000000    0.000000   0.0     0.0    0.0   
  # 3       135.0    0.000000    0.000000    0.000000  50.0  1404.0  790.0   
  # 4         0.0    0.000000    0.000000    0.000000   0.0     0.0    0.0   
  # ...       ...         ...         ...         ...   ...     ...    ...   
  # 590535    0.0   47.950001   47.950001   47.950001   0.0     0.0    0.0   
  # 590536    0.0    0.000000    0.000000    0.000000   0.0     0.0    0.0   
  # 590537    0.0    0.000000    0.000000    0.000000   0.0     0.0    0.0   
  # 590538  117.0  317.500000  669.500000  317.500000   0.0  2234.0    0.0   
  # 590539    0.0    0.000000    0.000000    0.000000   0.0     0.0    0.0   

  #               V319        V320        V321  V322  V323  V324  V325  V326  \
  # 0         0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   NaN   
  # 1         0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   NaN   
  # 2         0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   NaN   
  # 3         0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   NaN   
  # 4         0.000000    0.000000    0.000000   0.0   0.0   0.0   0.0   0.0   
  # ...            ...         ...         ...   ...   ...   ...   ...   ...   
  # 590535    0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   NaN   
  # 590536    0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   NaN   
  # 590537    0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   NaN   
  # 590538    0.000000    0.000000    0.000000   NaN   NaN   NaN   NaN   NaN   
  # 590539  279.950012  279.950012  279.950012   NaN   NaN   NaN   NaN   NaN   

  #         V327  V328  V329  V330  V331  V332  V333  V334  V335  V336  V337  \
  # 0        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 1        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 2        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 3        NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 4        0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   
  # ...      ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 590535   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590536   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590537   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590538   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
  # 590539   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   

  #         V338  V339  id_01    id_02  id_03  id_04  id_05  id_06  id_07  id_08  \
  # 0        NaN   NaN    NaN      NaN    NaN    NaN    NaN    NaN    NaN    NaN   
  # 1        NaN   NaN    NaN      NaN    NaN    NaN    NaN    NaN    NaN    NaN   
  # 2        NaN   NaN    NaN      NaN    NaN    NaN    NaN    NaN    NaN    NaN   
  # 3        NaN   NaN    NaN      NaN    NaN    NaN    NaN    NaN    NaN    NaN   
  # 4        0.0   0.0    0.0  70787.0    NaN    NaN    NaN    NaN    NaN    NaN   
  # ...      ...   ...    ...      ...    ...    ...    ...    ...    ...    ...   
  # 590535   NaN   NaN    NaN      NaN    NaN    NaN    NaN    NaN    NaN    NaN   
  # 590536   NaN   NaN    NaN      NaN    NaN    NaN    NaN    NaN    NaN    NaN   
  # 590537   NaN   NaN    NaN      NaN    NaN    NaN    NaN    NaN    NaN    NaN   
  # 590538   NaN   NaN    NaN      NaN    NaN    NaN    NaN    NaN    NaN    NaN   
  # 590539   NaN   NaN    NaN      NaN    NaN    NaN    NaN    NaN    NaN    NaN   

  #         id_09  id_10  id_11     id_12  id_13  id_14 id_15     id_16  id_17  \
  # 0         NaN    NaN    NaN       NaN    NaN    NaN   NaN       NaN    NaN   
  # 1         NaN    NaN    NaN       NaN    NaN    NaN   NaN       NaN    NaN   
  # 2         NaN    NaN    NaN       NaN    NaN    NaN   NaN       NaN    NaN   
  # 3         NaN    NaN    NaN       NaN    NaN    NaN   NaN       NaN    NaN   
  # 4         NaN    NaN  100.0  NotFound    NaN -480.0   New  NotFound  166.0   
  # ...       ...    ...    ...       ...    ...    ...   ...       ...    ...   
  # 590535    NaN    NaN    NaN       NaN    NaN    NaN   NaN       NaN    NaN   
  # 590536    NaN    NaN    NaN       NaN    NaN    NaN   NaN       NaN    NaN   
  # 590537    NaN    NaN    NaN       NaN    NaN    NaN   NaN       NaN    NaN   
  # 590538    NaN    NaN    NaN       NaN    NaN    NaN   NaN       NaN    NaN   
  # 590539    NaN    NaN    NaN       NaN    NaN    NaN   NaN       NaN    NaN   

  #         id_18  id_19  id_20  id_21  id_22 id_23  id_24  id_25  id_26 id_27  \
  # 0         NaN    NaN    NaN    NaN    NaN   NaN    NaN    NaN    NaN   NaN   
  # 1         NaN    NaN    NaN    NaN    NaN   NaN    NaN    NaN    NaN   NaN   
  # 2         NaN    NaN    NaN    NaN    NaN   NaN    NaN    NaN    NaN   NaN   
  # 3         NaN    NaN    NaN    NaN    NaN   NaN    NaN    NaN    NaN   NaN   
  # 4         NaN  542.0  144.0    NaN    NaN   NaN    NaN    NaN    NaN   NaN   
  # ...       ...    ...    ...    ...    ...   ...    ...    ...    ...   ...   
  # 590535    NaN    NaN    NaN    NaN    NaN   NaN    NaN    NaN    NaN   NaN   
  # 590536    NaN    NaN    NaN    NaN    NaN   NaN    NaN    NaN    NaN   NaN   
  # 590537    NaN    NaN    NaN    NaN    NaN   NaN    NaN    NaN    NaN   NaN   
  # 590538    NaN    NaN    NaN    NaN    NaN   NaN    NaN    NaN    NaN   NaN   
  # 590539    NaN    NaN    NaN    NaN    NaN   NaN    NaN    NaN    NaN   NaN   

  #       id_28     id_29        id_30                id_31  id_32      id_33  \
  # 0        NaN       NaN          NaN                  NaN    NaN        NaN   
  # 1        NaN       NaN          NaN                  NaN    NaN        NaN   
  # 2        NaN       NaN          NaN                  NaN    NaN        NaN   
  # 3        NaN       NaN          NaN                  NaN    NaN        NaN   
  # 4        New  NotFound  Android 7.0  samsung browser 6.2   32.0  2220x1080   
  # ...      ...       ...          ...                  ...    ...        ...   
  # 590535   NaN       NaN          NaN                  NaN    NaN        NaN   
  # 590536   NaN       NaN          NaN                  NaN    NaN        NaN   
  # 590537   NaN       NaN          NaN                  NaN    NaN        NaN   
  # 590538   NaN       NaN          NaN                  NaN    NaN        NaN   
  # 590539   NaN       NaN          NaN                  NaN    NaN        NaN   

  #                 id_34 id_35 id_36 id_37 id_38 DeviceType  \
  # 0                  NaN   NaN   NaN   NaN   NaN        NaN   
  # 1                  NaN   NaN   NaN   NaN   NaN        NaN   
  # 2                  NaN   NaN   NaN   NaN   NaN        NaN   
  # 3                  NaN   NaN   NaN   NaN   NaN        NaN   
  # 4       match_status:2     T     F     T     T     mobile   
  # ...                ...   ...   ...   ...   ...        ...   
  # 590535             NaN   NaN   NaN   NaN   NaN        NaN   
  # 590536             NaN   NaN   NaN   NaN   NaN        NaN   
  # 590537             NaN   NaN   NaN   NaN   NaN        NaN   
  # 590538             NaN   NaN   NaN   NaN   NaN        NaN   
  # 590539             NaN   NaN   NaN   NaN   NaN        NaN   

  #                           DeviceInfo  
  # 0                                 NaN  
  # 1                                 NaN  
  # 2                                 NaN  
  # 3                                 NaN  
  # 4       SAMSUNG SM-G892A Build/NRD90M  
  # ...                               ...  
  # 590535                            NaN  
  # 590536                            NaN  
  # 590537                            NaN  
  # 590538                            NaN  
  # 590539                            NaN  

  # [590540 rows x 434 columns]

  return merged

def check_nan(merged):
  number_of_rows_from_data=merged.shape[0]
  number_of_columns_from_data=merged.shape[1]

  # --------------------------------------------------------------------------------
  number_of_nan_in_column=merged.isnull().sum(axis=0)
  number_of_nan_in_row=merged.isnull().sum(axis=1)
  # print("number_of_nan_in_column",number_of_nan_in_column)
  # print("number_of_nan_in_row",number_of_nan_in_row)

  # --------------------------------------------------------------------------------
  # with pd.option_context('display.max_rows',100000):
  #   print("number_of_nan_in_column/number_of_rows_from_data*100",number_of_nan_in_column/number_of_rows_from_data*100)
  
  # with pd.option_context('display.max_rows',100000):
  #   print("number_of_nan_in_row/number_of_columns_from_data*100",number_of_nan_in_row/number_of_columns_from_data*100)

  # --------------------------------------------------------------------------------
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

  df=df.rename(columns={"index":'column_name',0:'nan_percent'})
  
  # ================================================================================
  fn='./results/nan_ratio_in_columns.csv'
  df.to_csv(fn,sep=',',encoding='utf-8',index=False)

def check_outlier(merged):
  print(merged.describe())

def create_normal_and_fraud_data(merged):
  normal_data=merged[merged['isFraud']==0]
  fraud_data=merged[merged['isFraud']==1]
  # print("normal_data",normal_data)
  # print("fraud_data",fraud_data)
  
  fn1='./results/merged_normal_full.csv'
  fn2='./results/merged_fraud_full.csv'
  normal_data.to_csv(fn1,sep=',',encoding='utf-8',index=False)
  fraud_data.to_csv(fn2,sep=',',encoding='utf-8',index=False)

def compare_each_item_between_0_and_1(merged_normal,merged_fraud):
  described_list=[]
  
  for i,one_column_name in tqdm(enumerate(list(merged_normal.columns))):

    if merged_normal[one_column_name].dtype==object:
      normal_object=merged_normal[one_column_name].value_counts(normalize = True).reset_index()
      fraud_object=merged_fraud[one_column_name].value_counts(normalize = True).reset_index()
      described_list.append([i,one_column_name,normal_object,fraud_object])
      # print('train_isFraud_value_counts',train_isFraud_value_counts)
      # 0    0.964904
      # 1    0.035096
      # 0    9650
      # 1     351
    else:
      normal_numeric=merged_normal[one_column_name].describe().reset_index()
      fraud_numeric=merged_fraud[one_column_name].describe().reset_index()
      normal_numeric=normal_numeric.drop(normal_numeric[normal_numeric['index']=="count"].index)
      fraud_numeric=fraud_numeric.drop(fraud_numeric[fraud_numeric['index']=="count"].index)
      normal_numeric=normal_numeric.drop(normal_numeric[normal_numeric['index']=="max"].index)
      fraud_numeric=fraud_numeric.drop(fraud_numeric[fraud_numeric['index']=="max"].index)
      described_list.append([i,one_column_name,normal_numeric,fraud_numeric])

  # print('described_list',described_list)
  # afaf
  
  for one_described in described_list:
    # print('one_described',one_described)
    
    # fig=plt.figure(figsize=(10,10))
    
    # plt.bar(list(one_described[0].iloc[:,0]),list(one_described[0].iloc[:,1]),
    #   color='red', # color
    #   alpha=0.5) # transparency
    # plt.show()

    x=1
    fig=plt.figure(figsize=(10,4))
    fig.suptitle(str(one_described[0]).zfill(4)+" "+one_described[1]+" : normal | fraud")

    for step in range(2):
      ax = fig.add_subplot(1, 2, x)
      # ax.set_title("fafaf", fontsize='large')
      plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.2)
      # ax.set_ylabel("mins", fontsize=12)

      import traceback
      try:
        # print('one_described',one_described)
        ax.bar(list(one_described[x+1].iloc[:,0]),list(one_described[x+1].iloc[:,1]), 0.35)
      except:
        # print('list(one_described[2].iloc[:,0])',list(one_described[2].iloc[:,0]))
        # print('list(one_described[3].iloc[:,1])',list(one_described[3].iloc[:,1]))
        import sys
        sys.exit(1)
      
      x=x+1

    # fig.tight_layout()
    # plt.show()
    plt.savefig("./results_images/"+str(one_described[0]).zfill(4)+"compare_"+one_described[1]+".png",dpi=400)
    plt.close()

def create_datetime_column(csv_train):
  # print('csv_train',csv_train)
  # csv_train
  #       TransactionID  isFraud  TransactionDT  TransactionAmt ProductCD  card1  \
  # 13285     2987031.00     0.00       86998.00          363.89         W   6573   
  # 483       2987111.00     0.00       88383.00           18.19         C  13832   

  start_datetime=datetime(2017,7,1,00,00)
  # start_datetime 2017-07-01 00:00:00

  converted_datetime_series=csv_train['TransactionDT'].map(lambda x:start_datetime+timedelta(seconds=x))
  # print('aa',aa)
  
  csv_train["TransactionDT_datetime"]=converted_datetime_series

  # ================================================================================
  rankdata_year_month=rankdata(list(csv_train["TransactionDT_datetime"].map(lambda x:str(x.year)+"-"+str(x.month).zfill(2))),method='dense')

  csv_train["TransactionDT_year_month"]=rankdata_year_month

  return csv_train

def visualize_number_of_data_in_each_second(csv_train,csv_test):

  # ================================================================================
  filtered_train_df=csv_train[["TransactionDT_datetime","TransactionID"]]
  filtered_train_df=filtered_train_df.set_index(filtered_train_df['TransactionDT_datetime'].apply(pd.to_datetime))
  aggregate_count_by_1second=filtered_train_df.resample('1m').count()
  # print('aggregate_count_by_1second',aggregate_count_by_1second.head())
  #                          TransactionDT_datetime  TransactionID
  # TransactionDT_datetime                                       
  # 2017-07-02 00:09:58                          1              1
  # 2017-07-02 00:09:59                          0              0
  # 2017-07-02 00:10:00                          0              0
  # 2017-07-02 00:10:01                          0              0
  # 2017-07-02 00:10:02                          0              0
  
  aggregate_count_by_1second_train=aggregate_count_by_1second[aggregate_count_by_1second['TransactionID']!=0]
  # print('pre_sale2',pre_sale2.shape)
  # pre_sale2 (19986, 2)

  # ================================================================================
  filtered_test_df=csv_test[["TransactionDT_datetime","TransactionID"]]
  filtered_test_df=filtered_test_df.set_index(filtered_test_df['TransactionDT_datetime'].apply(pd.to_datetime))
  aggregate_count_by_1second=filtered_test_df.resample('1m').count()
  # print('aggregate_count_by_1second',aggregate_count_by_1second.head())
  #                          TransactionDT_datetime  TransactionID
  # TransactionDT_datetime                                       
  # 2017-07-02 00:09:58                          1              1
  # 2017-07-02 00:09:59                          0              0
  # 2017-07-02 00:10:00                          0              0
  # 2017-07-02 00:10:01                          0              0
  # 2017-07-02 00:10:02                          0              0
  
  aggregate_count_by_1second_test=aggregate_count_by_1second[aggregate_count_by_1second['TransactionID']!=0]
  # print('aggregate_count_by_1second_test',aggregate_count_by_1second_test.shape)
  # aggregate_count_by_1second_test (12904, 2)

  # ================================================================================
  fig,ax=plt.subplots(1,1,figsize=(30,5))
  ax.plot(list(map(lambda x:str(x),list(aggregate_count_by_1second_train.index))),list(aggregate_count_by_1second_train["TransactionID"]),color="blue")
  ax.plot(list(map(lambda x:str(x),list(aggregate_count_by_1second_test.index))),list(aggregate_count_by_1second_test["TransactionID"]),color="red")
  

  # plt.show()
  fig.savefig('./results_images/time_in_train_and_test.png',dpi=1000)
  plt.close(fig)

def visualize_target_1_0_and_features(csv_train):
  # print('csv_train',csv_train)
  #        TransactionID  isFraud  TransactionDT  TransactionAmt ProductCD  card1  \
  # 13285     2987031.00     0.00       86998.00          363.89         W   6573   
  # 483       2987111.00     0.00       88383.00           18.19         C  13832   
  # 6239      2987261.00     0.00       90492.00           59.00         W  10049   
  # 15416     2987274.00     0.00       90715.00          222.00         W  10112   

  label_1_df=csv_train[csv_train['isFraud']==1]
  label_0_df=csv_train[csv_train['isFraud']==0]
  print("label_1_df",label_1_df)
  print("label_0_df",label_0_df)

def investigate_difference_of_target0_and_target1_numerical_date_distribution(csv_train):
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
  numerical_columns_all=numerical_columns_in_train_transaction+numerical_columns_in_train_identity
  categorical_columns_all=categorical_columns_in_train_transaction+categorical_columns_in_train_identity

  numerical_columns_all.remove('TransactionID')
  numerical_columns_all.remove('isFraud')
  numerical_columns_all.remove('TransactionDT')
  numerical_columns_all.remove('TransactionID')
  # print('numerical_columns_all',numerical_columns_all)

  # ================================================================================
  csv_train_class0=csv_train[csv_train['isFraud']==0]
  csv_train_class1=csv_train[csv_train['isFraud']==1]

  # ================================================================================
  for one_numerical_column in numerical_columns_all:

    lower_bound=csv_train[one_numerical_column].quantile(0.003)
    upper_bound=csv_train[one_numerical_column].quantile(0.997)
  
    csv_train_class0_filtered=csv_train_class0[one_numerical_column][
      (csv_train_class0[one_numerical_column]>=lower_bound)&
      (csv_train_class0[one_numerical_column]<=upper_bound)
    ]

    csv_train_class1_filtered=csv_train_class1[one_numerical_column][
      (csv_train_class1[one_numerical_column]>=lower_bound)&
      (csv_train_class1[one_numerical_column]<=upper_bound)
    ]

    # print('csv_train_class0_filtered',csv_train_class0_filtered)
    # print('csv_train_class1_filtered',csv_train_class1_filtered)
    csv_train_class0_filtered_hist,csv_train_class0_filtered_bin_edges=np.histogram(csv_train_class0_filtered)
    csv_train_class1_filtered_hist,csv_train_class1_filtered_bin_edges=np.histogram(csv_train_class1_filtered)
    # print("csv_train_class0_filtered_hist\n ",csv_train_class0_filtered_hist)
    # print("csv_train_class0_filtered_bin_edges\n",csv_train_class0_filtered_bin_edges)
    # print("csv_train_class1_filtered_hist\n",csv_train_class1_filtered_hist)
    # print("csv_train_class1_filtered_bin_edges\n",csv_train_class1_filtered_bin_edges)
    
    csv_train_class0_filtered_hist_normed=(np.array(csv_train_class0_filtered_hist)-np.array(csv_train_class0_filtered_hist).min())/(np.array(csv_train_class0_filtered_hist).max()-np.array(csv_train_class0_filtered_hist).min())
    csv_train_class1_filtered_hist_normed=(np.array(csv_train_class1_filtered_hist)-np.array(csv_train_class1_filtered_hist).min())/(np.array(csv_train_class1_filtered_hist).max()-np.array(csv_train_class1_filtered_hist).min())
    # print('normed',normed)
    # afaf

    fig,ax=plt.subplots(1,1,figsize=(20,5))
    ax.plot(csv_train_class0_filtered_bin_edges[:-1],csv_train_class0_filtered_hist_normed,color="blue")
    ax.plot(csv_train_class1_filtered_bin_edges[:-1],csv_train_class1_filtered_hist_normed,color="red")
    plt.close()
    # plt.show()
  
    # ================================================================================
    result_from_komogorov_smirnov=ks_2samp(csv_train_class0_filtered_hist_normed,csv_train_class1_filtered_hist_normed)
    # print('result_from_komogorov_smirnov',result_from_komogorov_smirnov)

    if result_from_komogorov_smirnov.pvalue<0.05:
      print('one_numerical_column',one_numerical_column)
      print('result_from_komogorov_smirnov',result_from_komogorov_smirnov.pvalue)
    # one_numerical_column C7
    # result_from_komogorov_smirnov 0.012340600575894691
    # one_numerical_column C12
    # result_from_komogorov_smirnov 0.002056766762649115
    # one_numerical_column V145
    # result_from_komogorov_smirnov 0.012340600575894691
    # one_numerical_column V160
    # result_from_komogorov_smirnov 0.002056766762649115
    # one_numerical_column V167
    # result_from_komogorov_smirnov 0.002056766762649115
    # one_numerical_column V177
    # result_from_komogorov_smirnov 0.002056766762649115
    # one_numerical_column V179
    # result_from_komogorov_smirnov 0.012340600575894691
    # one_numerical_column V186
    # result_from_komogorov_smirnov 0.012340600575894691
    # one_numerical_column V190
    # result_from_komogorov_smirnov 0.012340600575894691
    # one_numerical_column V193
    # result_from_komogorov_smirnov 0.012340600575894691
    # one_numerical_column V246
    # result_from_komogorov_smirnov 0.012340600575894691
    # one_numerical_column V257
    # result_from_komogorov_smirnov 0.012340600575894691
    # one_numerical_column V323
    # result_from_komogorov_smirnov 0.002056766762649115

def investigate_difference_of_target0_and_target1_categorical_date_distribution(csv_train):
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
  numerical_columns_all=numerical_columns_in_train_transaction+numerical_columns_in_train_identity
  categorical_columns_all=categorical_columns_in_train_transaction+categorical_columns_in_train_identity

  numerical_columns_all.remove('TransactionID')
  numerical_columns_all.remove('isFraud')
  numerical_columns_all.remove('TransactionDT')
  numerical_columns_all.remove('TransactionID')
  # print('numerical_columns_all',numerical_columns_all)

  # ================================================================================
  categorical_columns_all.append('TransactionID')
  csv_train_categorical=csv_train[categorical_columns_all]
  csv_train_categorical=csv_train_categorical.set_index("TransactionID")
  categorical_data_columns=list(csv_train_categorical.columns)
  
  label_encoder=preprocessing.LabelEncoder()
  csv_train_categorical[categorical_data_columns]=csv_train_categorical[categorical_data_columns].apply(lambda x:label_encoder.fit_transform(x.astype(str)))
  # print('csv_train_categorical',csv_train_categorical)
  #                ProductCD  card1  card2  card3  card4  card5  card6  addr1  \
  # TransactionID                                                               
  # 2987031.00             4   2837    468     23      4     53      0     52   
  # 2987111.00             0    834    267     37      2     51      1    104   
  
  merged=pd.merge(csv_train_categorical.reset_index(),csv_train[["TransactionID","isFraud"]],on=['TransactionID'],how='left')
  # print('merged',merged)

  # ================================================================================
  csv_train_class0=merged[merged['isFraud']==0]
  csv_train_class1=merged[merged['isFraud']==1]
  # print("csv_train_class0",csv_train_class0)
  # print("csv_train_class1",csv_train_class1)
  
  categorical_columns_all=list(csv_train_class0.columns)
  categorical_columns_all.remove("TransactionID")

  # ================================================================================
  for one_categorical_column in categorical_columns_all:

    csv_train_class0_filtered=csv_train_class0[[one_categorical_column]]
    csv_train_class1_filtered=csv_train_class1[[one_categorical_column]]
    # print("one_categorical_column",one_categorical_column)
    # print("csv_train_class0_filtered",csv_train_class0_filtered)
    # print("csv_train_class1_filtered",csv_train_class1_filtered)

    csv_train_class0_filtered_hist,csv_train_class0_filtered_bin_edges=np.histogram(csv_train_class0_filtered)
    csv_train_class1_filtered_hist,csv_train_class1_filtered_bin_edges=np.histogram(csv_train_class1_filtered)
    # print("csv_train_class0_filtered_hist\n ",csv_train_class0_filtered_hist)
    # print("csv_train_class0_filtered_bin_edges\n",csv_train_class0_filtered_bin_edges)
    # print("csv_train_class1_filtered_hist\n",csv_train_class1_filtered_hist)
    # print("csv_train_class1_filtered_bin_edges\n",csv_train_class1_filtered_bin_edges)
    
    csv_train_class0_filtered_hist_normed=(np.array(csv_train_class0_filtered_hist)-np.array(csv_train_class0_filtered_hist).min())/(np.array(csv_train_class0_filtered_hist).max()-np.array(csv_train_class0_filtered_hist).min())
    csv_train_class1_filtered_hist_normed=(np.array(csv_train_class1_filtered_hist)-np.array(csv_train_class1_filtered_hist).min())/(np.array(csv_train_class1_filtered_hist).max()-np.array(csv_train_class1_filtered_hist).min())
    # print('normed',normed)
    # afaf

    fig,ax=plt.subplots(1,1,figsize=(20,5))
    ax.plot(csv_train_class0_filtered_bin_edges[:-1],csv_train_class0_filtered_hist_normed,color="blue")
    ax.plot(csv_train_class1_filtered_bin_edges[:-1],csv_train_class1_filtered_hist_normed,color="red")
    plt.close()
    # plt.show()
  
    # ================================================================================
    result_from_komogorov_smirnov=ks_2samp(csv_train_class0_filtered_hist_normed,csv_train_class1_filtered_hist_normed)
    # print('result_from_komogorov_smirnov',result_from_komogorov_smirnov)

    if result_from_komogorov_smirnov.pvalue<0.05:
      print('one_categorical_column',one_categorical_column)
      print('result_from_komogorov_smirnov',result_from_komogorov_smirnov.pvalue)
    # one_categorical_column addr1
    # result_from_komogorov_smirnov 0.012340600575894691

def check_time_consistency_of_each_feature(csv_train):
  # print('csv_train',list(csv_train.columns))
  # ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181', 'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253', 'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321', 'V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'TransactionDT_datetime', 'TransactionDT_year_month']

  feature_to_be_checked=list(csv_train.columns)
  feature_to_be_checked.remove("isFraud")
  feature_to_be_checked.remove("TransactionDT_datetime")
  feature_to_be_checked.remove("TransactionDT_year_month")

  # print('csv_train',csv_train)
  csv_train=csv_train.set_index("TransactionID")
  csv_train_target_df=csv_train[["isFraud","TransactionDT_year_month"]]

  feature_to_be_checked.remove("TransactionID")
  
  all_features_name_for_visualizeation=[]
  all_features_rocauc_for_visualizeation=[]
  for one_feature_to_be_checked in feature_to_be_checked:
    one_feature_df=csv_train[[one_feature_to_be_checked]].reset_index()
    one_feature_df=pd.merge(one_feature_df,csv_train_target_df,on=['TransactionID'],how='left')
    # print('one_feature_df',one_feature_df)
    # one_feature_df
    #       TransactionID  TransactionDT  isFraud  TransactionDT_year_month
    # 0         2987031.00       86998.00     0.00                         1
    # 1         2987111.00       88383.00     0.00                         1
    # 2         2987261.00       90492.00     0.00                         1
    # 3         2987274.00       90715.00     0.00                         1
    # 4         2987285.00       90892.00     0.00                         1
    # ...              ...            ...      ...                       ...
    # 19996     3577426.00    15809098.00     0.00                         6
    # 19997     3577436.00    15809269.00     0.00                         6
    # 19998     3577443.00    15809412.00     0.00                         6
    # 19999     3577451.00    15809554.00     0.00                         6
    # 20000     3577466.00    15809796.00     0.00                         6

    # [20001 rows x 4 columns]
    
    # ================================================================================
    normalized_train_X=one_feature_df.iloc[:,1].reset_index()
    train_y=one_feature_df.iloc[:,2].reset_index()
    del normalized_train_X["index"]
    del train_y["index"]
    # print("normalized_train_X",type(normalized_train_X))
    # print("train_y",type(train_y))

    # # ================================================================================
    # def objective(trial):

    #   # ================================================================================
    #   group_kfold=GroupKFold(n_splits=4)
    #   groups=list(one_feature_df['TransactionDT_year_month'])

    #   roc_auc_score_init=0
    #   for fold_n,(train,test) in enumerate(group_kfold.split(normalized_train_X,train_y,groups)):
    #     # print("Fold : ",fold_n)

    #     X_train_,X_valid=normalized_train_X.iloc[train],normalized_train_X.iloc[test]
    #     y_train_,y_valid=train_y.iloc[train],train_y.iloc[test]

    #     # ================================================================================
    #     parameters = {
    #       'n_estimators': trial.suggest_int("n_estimators", 80, 401, step=10),
    #       'learning_rate': trial.suggest_float("learning_rate", low=0.05, high=0.46, step=0.05),
    #       'num_leaves': trial.suggest_int("num_leaves", low=2, high=101, step=10),
    #       'max_depth': trial.suggest_int("max_depth", low=-1, high=41, step=4),  
    #       'boosting': trial.suggest_categorical("boosting", choices=['gbdt', 'dart']) 
    #     }
        
    #     lgbclf=lgb.LGBMClassifier(**parameters)

    #     # ================================================================================
    #     # Train lgb model with train dataset

    #     lgbclf.fit(X_train_,y_train_.values.ravel())
        
    #     # ================================================================================
    #     # Delete used data

    #     del X_train_,y_train_

    #     # ================================================================================
    #     # Make prediction on test dataset

    #     val=lgbclf.predict_proba(X_valid)[:,1]
    #     # print("pred",pred)

    #     # ================================================================================
    #     # Delete used data

    #     del X_valid

    #     # ================================================================================
    #     roc_auc_score_init+=roc_auc_score(y_valid,val)/4

    #   return roc_auc_score_init

    # # ================================================================================
    # default_params = {'n_estimators': 100, 
    #                   'learning_rate': 0.1, 
    #                   'num_leaves': 31, 
    #                   'max_depth': -1, 
    #                   'boosting': 'gbdt'}
    # study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
    # study.enqueue_trial(default_params)
    # study.optimize(objective, n_trials=100)
    # best_trial = study.best_trial
    # print('Best parameters LightGBM:', best_trial.params)

    # ================================================================================
    group_kfold=GroupKFold(n_splits=4)
    groups=list(one_feature_df['TransactionDT_year_month'])

    roc_auc_score_init=0
    for fold_n,(train,test) in enumerate(group_kfold.split(normalized_train_X,train_y,groups)):
      # print("Fold : ",fold_n)

      X_train_,X_valid=normalized_train_X.iloc[train],normalized_train_X.iloc[test]
      y_train_,y_valid=train_y.iloc[train],train_y.iloc[test]

      # ================================================================================
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

      params={
        'n_estimators':100, 
        'learning_rate':0.1, 
        'num_leaves':31, 
        'max_depth':-1, 
        'boosting':'gbdt'
      }

      # ================================================================================
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
      roc_auc_score_init+=roc_auc_score(y_valid,val)/4

    if roc_auc_score_init<0.5:
      print('one_feature_to_be_checked',one_feature_to_be_checked)
      print('roc_auc_score_init',roc_auc_score_init)

    all_features_name_for_visualizeation.append(one_feature_to_be_checked)
    all_features_rocauc_for_visualizeation.append(roc_auc_score_init)

  with open('./pickles/all_features_name_for_visualizeation.pkl','wb') as f:
    pickle.dump(all_features_name_for_visualizeation,f)
  with open('./pickles/all_features_name_for_visualizeation.pkl','rb') as f:
    all_features_name_for_visualizeation=pickle.load(f)
  
  with open('./pickles/all_features_rocauc_for_visualizeation.pkl','wb') as f:
    pickle.dump(all_features_rocauc_for_visualizeation,f)
  with open('./pickles/all_features_rocauc_for_visualizeation.pkl','rb') as f:
    all_features_rocauc_for_visualizeation=pickle.load(f)

  # fig,ax=plt.subplots(1,1,figsize=(20,5))
  fig,ax=plt.subplots(1,1)
  ax.bar(all_features_name_for_visualizeation,all_features_rocauc_for_visualizeation)
  ax.set_title('Time consistency test on all features')
  ax.set_xlabel('Feature names')
  ax.set_ylabel('ROC AUC score (average from 4 groupkfolds)')
  ax.set_xticklabels(all_features_name_for_visualizeation,rotation=90,fontsize=2)
  # plt.show()
  ax.axhline(y=0.5,color='r',linestyle='--')
  ax.axhline(y=0.52,color='r',linestyle='--')
  fig.savefig('./results_images/time_consistency_result.png',dpi=2000)
  plt.close(fig)
  afaf

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

# ================================================================================
# Load csv file

csv_train=load_csv_files_train()
csv_test=load_csv_files_test()
# print("csv_train",csv_train)
# print("csv_test",csv_test)

# ================================================================================
train_nan_ratio_df,train_columns_to_be_dropped=check_nan(csv_train,NAN_CRITERION,create_image=CREATE_IMAGE_ON_NAN_RATIO)

# ================================================================================
# Discard columns whose NaNs are too abundant

csv_train=discard_nan_columns(csv_train,train_columns_to_be_dropped)
# print('csv_train',csv_train)

# ================================================================================
# Set datetime index

csv_train=create_datetime_column(csv_train)
csv_test=create_datetime_column(csv_test)

# ================================================================================
# csv_train_test=pd.concat([csv_train,csv_test])

# visualize_number_of_data_in_each_second(csv_train,csv_test)

# ================================================================================
# visualize_target_1_0_and_features(csv_train)

# ================================================================================
# investigate_difference_of_target0_and_target1_numerical_date_distribution(csv_train)

# ================================================================================
# investigate_difference_of_target0_and_target1_categorical_date_distribution(csv_train)

# ================================================================================
# Check time consistency of each feature

check_time_consistency_of_each_feature(csv_train)
afaf 



# --------------------------------------------------------------------------------
# Load normal and fraud csv file

# merged_normal,merged_fraud=load_normal_fraud_csv_files()

# --------------------------------------------------------------------------------
# Merge transaction and identity data

# merged=merge_transaction_identity(csv_train_identity,csv_train_transaction_original)

# --------------------------------------------------------------------------------
# check NaN in column and row

# check_nan(merged)

# --------------------------------------------------------------------------------
# check outliers in column

# check_outlier(merged)

# --------------------------------------------------------------------------------
# Create normal and fraud data

# create_normal_and_fraud_data(merged)

# --------------------------------------------------------------------------------
# Compare each item between 0 and 1

# compare_each_item_between_0_and_1(merged_normal,merged_fraud)

# --------------------------------------------------------------------------------
# Compare TransactionAmt between 0 and 1

