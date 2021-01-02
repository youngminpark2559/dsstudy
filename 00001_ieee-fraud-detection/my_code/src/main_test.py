# conda activate py36_django_bare && \
# cd /mnt/external_disk/Companies/side_project/Kaggle/00001_ieee-fraud-detection/my_code/src && \
# rm e.l && python main_test.py \
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

# ================================================================================
from sklearn.utils import resample
import gc
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from hyperopt import hp,tpe,fmin
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split,cross_val_score
import pickle
from datetime import datetime,timedelta,date
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.float_format','{:.2f}'.format)

# ================================================================================
# OPTIONS

TRAIN_DATA_SIZE="full"
# TRAIN_DATA_SIZE="small"

# CREATE_IMAGE_ON_NAN_RATIO=True
CREATE_IMAGE_ON_NAN_RATIO=False

# METHOD_FOR_IMPUTE_NUMERICAL_DATA="mice"
METHOD_FOR_IMPUTE_NUMERICAL_DATA="mean"

IMPUTED_NUMERICAL_DATA_SOURCE="function"
# IMPUTED_NUMERICAL_DATA_SOURCE="pickle"

ENCODED_CATEGORICAL_DATA_SOURCE="function"
# ENCODED_CATEGORICAL_DATA_SOURCE="pickle"

# RESAMPLING="oversampling_smote"
RESAMPLING="oversampling_resampling"
# RESAMPLING="undersampling_resampling"

# HYPERPARAMETER_TUNIING_LGBM=True
HYPERPARAMETER_TUNIING_LGBM=False

# HYPERPARAMETER_TUNIING_LGBM_USE=True
HYPERPARAMETER_TUNIING_LGBM_USE=False

# PERFORM_FULL_VALIDATION=True
PERFORM_FULL_VALIDATION=False

PERFORM_FULL_TRAIN=True
# PERFORM_FULL_TRAIN=False

DEEP_LEARNING_EPOCH=1000

# ================================================================================
# def load_csv_files():
#   # csv_train_transaction_original=pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/test_transaction.csv',encoding='utf-8')
#   csv_train_transaction_original=pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/test_transaction_original.csv',encoding='utf-8')
#   return csv_train_transaction_original

def load_csv_files():
  
  # ================================================================================
  # COLUMNS WITH STRINGS

  str_type = ['ProductCD', 'card1',  'card2', 'card3', 'card4', 'card5', 'card6',
              'addr1', 'addr2',
              'P_emaildomain', 'R_emaildomain','M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
              'id-12', 'id-14', 'id-15', 'id-16', 'id-21', 'id-22', 'id-23', 'id-24', 'id-25', 'id-26', 'id-27', 'id-28', 'id-29', 'id-30', 'id-31', 'id-32', 'id-33', 'id-34', 'id-35', 'id-36', 'id-37', 'id-38', 
              'DeviceType', 'DeviceInfo']

  # str_type += ['id-12', 'id-15', 'id-16', 'id-23', 'id-27', 'id-28', 'id-29', 'id-30', 
  #             'id-31', 'id-33', 'id-34', 'id-35', 'id-36', 'id-37', 'id-38']

  # ================================================================================
  # FIRST 53 COLUMNS

  id_numeric=["id-"+str(i).zfill(2) for i in range(1,12)]

  cols = ['TransactionID', 'TransactionDT', 'TransactionAmt', 'dist1', 'dist2', 
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
        'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
        'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15']+id_numeric

  # ================================================================================
  # V COLUMNS TO LOAD DECIDED BY CORRELATION EDA
  # https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id

  v =  [1, 3, 4, 6, 8, 11]
  v += [13, 14, 17, 20, 23, 26, 27, 30]
  v += [36, 37, 40, 41, 44, 47, 48]
  v += [54, 56, 59, 62, 65, 67, 68, 70]
  v += [76, 78, 80, 82, 86, 88, 89, 91]

  #v += [96, 98, 99, 104] #relates to groups, no NAN 
  v += [107, 108, 111, 115, 117, 120, 121, 123] # maybe group, no NAN
  v += [124, 127, 129, 130, 136] # relates to groups, no NAN

  # LOTS OF NAN BELOW
  v += [138, 139, 142, 147, 156, 162] #b1
  v += [165, 160, 166] #b1
  v += [178, 176, 173, 182] #b2
  v += [187, 203, 205, 207, 215] #b2
  v += [169, 171, 175, 180, 185, 188, 198, 210, 209] #b2
  v += [218, 223, 224, 226, 228, 229, 235] #b3
  v += [240, 258, 257, 253, 252, 260, 261] #b3
  v += [264, 266, 267, 274, 277] #b3
  v += [220, 221, 234, 238, 250, 271] #b3

  v += [294, 284, 285, 286, 291, 297] # relates to grous, no NAN
  v += [303, 305, 307, 309, 310, 320] # relates to groups, no NAN
  v += [281, 283, 289, 296, 301, 314] # relates to groups, no NAN
  #v += [332, 325, 335, 338] # b4 lots NAN

  # print('v',v)
  # v [1, 3, 4, 6, 8, 11, 13, 14, 17, 20, 23, 26, 27, 30, 36, 37, 40, 41, 44, 47, 48, 54, 56, 59, 62, 65, 67, 68, 70, 76, 78, 80, 82, 86, 88, 89, 91, 107, 108, 111, 115, 117, 120, 121, 123, 124, 127, 129, 130, 136, 138, 139, 142, 147, 156, 162, 165, 160, 166, 178, 176, 173, 182, 187, 203, 205, 207, 215, 169, 171, 175, 180, 185, 188, 198, 210, 209, 218, 223, 224, 226, 228, 229, 235, 240, 258, 257, 253, 252, 260, 261, 264, 266, 267, 274, 277, 220, 221, 234, 238, 250, 271, 294, 284, 285, 286, 291, 297, 303, 305, 307, 309, 310, 320, 281, 283, 289, 296, 301, 314]

  # ================================================================================
  cols += ['V'+str(x) for x in v]

  dtypes = {}
  for c in cols:
      dtypes[c] = 'float32'

  for c in str_type:
      dtypes[c] = 'category'

  # print('dtypes',dtypes)

  use_cols=cols+str_type

  # ================================================================================
  # Load train data

  # print('dtypes',dtypes)

  if TRAIN_DATA_SIZE=="full":
    # X_train = pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/train_transaction_original.csv', dtype=dtypes, usecols=use_cols+['isFraud'])
    X_train = pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/test_transaction_original.csv', dtype=dtypes,)

  elif TRAIN_DATA_SIZE=="small":

    # X_train = pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/train_transaction_small.csv', dtype=dtypes, usecols=use_cols+['isFraud'])
    X_train = pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/test_transaction.csv', dtype=dtypes)

  train_id = pd.read_csv('/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/test_identity.csv', dtype=dtypes)

  # X_train = X_train.merge(train_id, how='left', left_index=True, right_index=True)
  X_train=pd.merge(X_train,train_id,on=['TransactionID'],how='left')

  X_train=X_train.sort_values(by=['TransactionID'],axis=0)
  # print('X_train',X_train)

  # ================================================================================
  # Select columns

  X_train=X_train[use_cols]
  # print('X_train',X_train)

  # ================================================================================
  # Remove columns

  for c in ['D6','D7','D8','D9','D12','D13','D14','C3','M5','id-08','id-33','card4','id-07','id-14','id-21','id-30','id-32','id-34']+['id-'+str(x) for x in range(22,28)]:
    del X_train[c]

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

def check_nan(merged,create_image):
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

  df=df.rename(columns={"index":'column_name',0:'nan_percent'})

  if create_image==True:
    plt.bar(list(df["column_name"]),list(df["nan_percent"]))
    plt.xticks(rotation=90,fontsize=0.3)
    # plt.show()
    plt.savefig('aa.png',dpi=4000)
    # /mnt/external_disk/Companies/side_project/Kaggle/00001_ieee-fraud-detection/my_code/src/aa.png
  
  return df

def discard_40percent_nan_columns(merged):

  with open('./pickles/under40_nan_df_columns_name.pkl','rb') as f:
    under40_nan_df_columns_name=pickle.load(f)
    under40_nan_df_columns_name.remove("isFraud")
  with open('./pickles/under40_nan_df_numerical_columns_name.pkl','rb') as f:
    under40_nan_df_numerical_columns_name=pickle.load(f)
  with open('./pickles/under40_nan_df_categorical_columns_name.pkl','rb') as f:
    under40_nan_df_categorical_columns_name=pickle.load(f)
  
  under40_nan_df=merged[list(under40_nan_df_columns_name)]
  # print('under40_nan_df',under40_nan_df)
  # print('under40_nan_df_numerical_columns_name',under40_nan_df_numerical_columns_name)
  # print('under40_nan_df_categorical_columns_name',under40_nan_df_categorical_columns_name)
  # under40_nan_df_numerical_columns_name ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D4', 'D10', 'D15', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321']
  # under40_nan_df_categorical_columns_name ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'M6']

  return under40_nan_df,under40_nan_df_categorical_columns_name,under40_nan_df_numerical_columns_name

def separate_full_column_data_into_categorical_and_numerical(csv_train):

  # with pd.option_context('display.max_rows',100000):
  #   print('csv_train',csv_train.dtypes)

  csv_train=csv_train.set_index("TransactionID")
  # print('csv_train',csv_train)

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

  numerical_df=pd.concat(numerical_data,axis=1)
  categorical_df=pd.concat(categorical_data,axis=1)
  # print("numerical_df",numerical_df)
  # print("categorical_df",categorical_df)

  return numerical_df,categorical_df

def impute_categorical_data_by_mode(under40_nan_categorical_df):
  # print('under40_nan_categorical_df',under40_nan_categorical_df)
  #        TransactionID ProductCD  card1  card2  card3  card5   card6  addr1  \
  # 0         2987031.00         W   6573  583.0  150.0  226.0  credit  315.0   
  # 1         2987111.00         C  13832  375.0  185.0  224.0   debit    NaN   
  # 2         2987261.00         W  10049  555.0  150.0  226.0   debit  191.0   
  # 3         2987274.00         W  10112  360.0  150.0  166.0   debit  512.0   
  # 4         2987285.00         W  10520  543.0  150.0  224.0   debit  299.0   
  # ...              ...       ...    ...    ...    ...    ...     ...    ...   
  # 19996     3577426.00         W   7239  452.0  150.0  117.0   debit  264.0   
  # 19997     3577436.00         W  10975  555.0  150.0  117.0   debit  264.0   
  # 19998     3577443.00         W   1431  492.0  150.0  226.0   debit  343.0   
  # 19999     3577451.00         R  16434  399.0  150.0  146.0  credit  299.0   
  # 20000     3577466.00         W   7815  161.0  150.0  117.0   debit  123.0   

  #       addr2 dist1  dist2 P_emaildomain R_emaildomain   M1   M2   M3   M4   M6  \
  # 0      87.0  13.0    NaN     yahoo.com           NaN    T    T    T  NaN    T   
  # 1       NaN   NaN  109.0   hotmail.com   hotmail.com  NaN  NaN  NaN   M0  NaN   
  # 2      87.0   8.0    NaN           NaN           NaN    T    T    T   M1    T   
  # 3      87.0   NaN    NaN   hotmail.com           NaN  NaN  NaN  NaN   M0    F   
  # 4      87.0   2.0    NaN     gmail.com           NaN    T    T    T   M0    T   
  # ...     ...   ...    ...           ...           ...  ...  ...  ...  ...  ...   
  # 19996  87.0   8.0    NaN     gmail.com           NaN    T    T    T  NaN    F   
  # 19997  87.0   NaN    NaN     gmail.com           NaN    T    T    T   M0    F   
  # 19998  87.0   8.0    NaN       aol.com           NaN    T    T    T  NaN    F   
  # 19999  87.0   NaN   40.0     gmail.com     gmail.com  NaN  NaN  NaN  NaN  NaN   
  # 20000  87.0  22.0    NaN     yahoo.com           NaN    T    F    F   M0    T   

  #         M7   M8   M9     id_12 id_15     id_16  id_28     id_29        id_31  \
  # 0        F    T    T       NaN   NaN       NaN    NaN       NaN          NaN   
  # 1      NaN  NaN  NaN  NotFound   New  NotFound  Found     Found  chrome 62.0   
  # 2      NaN  NaN  NaN       NaN   NaN       NaN    NaN       NaN          NaN   
  # 3      NaN  NaN  NaN       NaN   NaN       NaN    NaN       NaN          NaN   
  # 4        F    T    T       NaN   NaN       NaN    NaN       NaN          NaN   
  # ...    ...  ...  ...       ...   ...       ...    ...       ...          ...   
  # 19996    F    T    T       NaN   NaN       NaN    NaN       NaN          NaN   
  # 19997  NaN  NaN  NaN       NaN   NaN       NaN    NaN       NaN          NaN   
  # 19998    F    F    T       NaN   NaN       NaN    NaN       NaN          NaN   
  # 19999  NaN  NaN  NaN  NotFound   New  NotFound    New  NotFound       google   
  # 20000    F    F    F       NaN   NaN       NaN    NaN       NaN          NaN   

  #       id_35 id_36 id_37 id_38 DeviceType  DeviceInfo  card1_addr1  \
  # 0       NaN   NaN   NaN   NaN        NaN         NaN   6573_315.0   
  # 1         F     F     T     T    desktop     Windows    13832_nan   
  # 2       NaN   NaN   NaN   NaN        NaN         NaN  10049_191.0   
  # 3       NaN   NaN   NaN   NaN        NaN         NaN  10112_512.0   
  # 4       NaN   NaN   NaN   NaN        NaN         NaN  10520_299.0   
  # ...     ...   ...   ...   ...        ...         ...          ...   
  # 19996   NaN   NaN   NaN   NaN        NaN         NaN   7239_264.0   
  # 19997   NaN   NaN   NaN   NaN        NaN         NaN  10975_264.0   
  # 19998   NaN   NaN   NaN   NaN        NaN         NaN   1431_343.0   
  # 19999     T     F     T     F     mobile  iOS Device  16434_299.0   
  # 20000   NaN   NaN   NaN   NaN        NaN         NaN   7815_123.0   

  #       card1_addr1_P_emaildomain  
  # 0          6573_315.0_yahoo.com  
  # 1         13832_nan_hotmail.com  
  # 2               10049_191.0_nan  
  # 3       10112_512.0_hotmail.com  
  # 4         10520_299.0_gmail.com  
  # ...                         ...  
  # 19996      7239_264.0_gmail.com  
  # 19997     10975_264.0_gmail.com  
  # 19998        1431_343.0_aol.com  
  # 19999     16434_299.0_gmail.com  
  # 20000      7815_123.0_yahoo.com  

  # [20001 rows x 35 columns]

  # ================================================================================
  # Find mode

  temp_df1=[]
  # under40_nan_categorical_df_mode=under40_nan_categorical_df.mode().astype(str)
  for one_column_name in under40_nan_categorical_df:

    # one_column_data=under40_nan_categorical_df[[one_column_name]]
    # bb=under40_nan_categorical_df[[one_column_name]].fillna(one_column_data.value_counts().index[0][0])

    under40_nan_categorical_df[one_column_name]=under40_nan_categorical_df[one_column_name].cat.add_categories('nullstr')
    bb=under40_nan_categorical_df[[one_column_name]].fillna("nullstr")

    temp_df1.append(bb)
  
  temp_df2=pd.concat(temp_df1,axis=1)
  # print('temp_df2',temp_df2)
 
  return temp_df2


def impute_numerical_data_by_MICE(under40_nan_numerical_df):
  # print('under40_nan_numerical_df',under40_nan_numerical_df)
  #        TransactionID  isFraud  TransactionDT  TransactionAmt     C1     C2  \
  # 0            3457624        0       12153579         724.000    3.0    1.0   
  # 1            3552820        0       15005886         108.500    2.0    1.0   
  # 2            3271083        0        6970178          47.950    1.0    1.0   
  # 3            3226689        0        5673658         100.599    2.0    3.0   
  # 4            3268855        0        6886780         107.950   10.0   14.0   
  # ...              ...      ...            ...             ...    ...    ...   
  # 19996        3131917        0        2993343          57.950  146.0  134.0   
  # 19997        3166342        0        3942261         335.000    4.0    2.0   
  # 19998        3382385        0        9933523         107.950  153.0  148.0   
  # 19999        3082613        0        1964836         150.000    3.0    2.0   
  # 20000        3554637        0       15041881          21.702    4.0    8.0   

  #         C3   C4     C5     C6   C7   C8    C9  C10    C11  C12    C13    C14  \
  # 0      0.0  0.0    0.0    1.0  0.0  0.0   2.0  0.0    1.0  0.0    2.0    2.0   
  # 1      0.0  0.0    0.0    1.0  0.0  0.0   1.0  0.0    1.0  1.0    7.0    2.0   
  # 2      0.0  0.0    2.0    1.0  0.0  0.0   1.0  0.0    1.0  0.0    3.0    1.0   
  # 3      0.0  1.0    0.0    1.0  1.0  0.0   0.0  0.0    1.0  1.0    0.0    0.0   
  # 4      0.0  0.0   11.0    8.0  0.0  0.0   6.0  0.0   10.0  0.0   43.0    9.0   
  # ...    ...  ...    ...    ...  ...  ...   ...  ...    ...  ...    ...    ...   
  # 19996  0.0  0.0  124.0   96.0  0.0  0.0  77.0  0.0  102.0  0.0  533.0  122.0   
  # 19997  0.0  0.0    0.0    2.0  0.0  0.0   2.0  0.0    1.0  0.0   47.0    4.0   
  # 19998  0.0  0.0  127.0  114.0  0.0  0.0  93.0  0.0  116.0  0.0  480.0  128.0   
  # 19999  0.0  1.0    0.0    1.0  0.0  2.0   0.0  2.0    2.0  0.0    2.0    2.0   
  # 20000  0.0  1.0    0.0    1.0  1.0  1.0   0.0  2.0    1.0  1.0    0.0    0.0   

  #           D1     D4    D10    D15  V12  V13  V14  V15  V16  V17  V18  V19  \
  # 0        0.0  145.0  145.0  145.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0   
  # 1      122.0  347.0  347.0  347.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0   
  # 2       32.0   22.0   33.0   33.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0   
  # 3        0.0    0.0    0.0    0.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  1.0   
  # 4      549.0  549.0  446.0  549.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0   
  # ...      ...    ...    ...    ...  ...  ...  ...  ...  ...  ...  ...  ...   
  # 19996   99.0  307.0    0.0    0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0   
  # 19997   50.0  509.0  518.0  518.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0   
  # 19998  578.0  590.0  590.0  590.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0   
  # 19999    0.0  365.0  365.0  365.0  0.0  0.0  1.0  0.0  0.0  1.0  1.0  1.0   
  # 20000    0.0   45.0    0.0   45.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0   

  #       V20  V21  V22  V23  V24  V25  V26  V27  V28  V29  V30  V31  V32  V33  \
  # 0      1.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   
  # 1      1.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   
  # 2      0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0   
  # 3      1.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0   
  # 4      1.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0   
  # ...    ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...   
  # 19996  1.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0   
  # 19997  1.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   
  # 19998  1.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0   
  # 19999  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0   
  # 20000  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0   

  #       V34  V35  V36  V37  V38  V39  V40  V41  V42  V43  V44  V45  V46  V47  \
  # 0      0.0  1.0  1.0  1.0  1.0  0.0  0.0  1.0  0.0  0.0  1.0  1.0  1.0  1.0   
  # 1      0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  0.0  0.0  1.0  1.0  1.0  1.0   
  # 2      0.0  1.0  1.0  1.0  1.0  0.0  0.0  1.0  0.0  0.0  1.0  1.0  1.0  1.0   
  # 3      1.0  0.0  0.0  3.0  3.0  0.0  0.0  1.0  0.0  0.0  2.0  2.0  1.0  1.0   
  # 4      0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  0.0  0.0  1.0  1.0  1.0  1.0   
  # ...    ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...   
  # 19996  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  0.0  0.0  1.0  1.0  1.0  1.0   
  # 19997  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  0.0  0.0  1.0  2.0  1.0  1.0   
  # 19998  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  0.0  0.0  1.0  1.0  1.0  1.0   
  # 19999  0.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0   
  # 20000  1.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0   

  #       V48  V49  V50  V51  V52  V53  V54  V55  V56  V57  V58  V59  V60  V61  \
  # 0      0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0   
  # 1      0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0   
  # 2      1.0  1.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0   
  # 3      0.0  0.0  1.0  1.0  1.0  0.0  0.0  2.0  2.0  1.0  1.0  0.0  0.0  1.0   
  # 4      1.0  1.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0   
  # ...    ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...   
  # 19996  1.0  1.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0   
  # 19997  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0   
  # 19998  1.0  1.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0   
  # 19999  0.0  0.0  1.0  0.0  0.0  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   
  # 20000  0.0  0.0  1.0  1.0  1.0  0.0  0.0  5.0  5.0  1.0  1.0  1.0  1.0  1.0   

  #       V62  V63  V64  V65  V66  V67  V68  V69  V70  V71  V72  V73  V74  V75  \
  # 0      1.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   
  # 1      1.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   
  # 2      0.0  0.0  0.0  1.0  1.0  1.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0   
  # 3      1.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0   
  # 4      1.0  0.0  0.0  1.0  1.0  1.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0   
  # ...    ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...   
  # 19996  1.0  0.0  0.0  1.0  1.0  1.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0   
  # 19997  1.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   
  # 19998  1.0  0.0  0.0  1.0  1.0  1.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0   
  # 19999  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  0.0   
  # 20000  1.0  4.0  4.0  1.0  1.0  1.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0   

  #       V76  V77  V78  V79  V80  V81  V82  V83  V84  V85  V86  V87  V88  V89  \
  # 0      1.0  1.0  1.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0   
  # 1      0.0  1.0  1.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0   
  # 2      1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  0.0   
  # 3      0.0  3.0  3.0  1.0  0.0  0.0  1.0  1.0  0.0  0.0  2.0  2.0  1.0  0.0   
  # 4      0.0  1.0  1.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0   
  # ...    ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...   
  # 19996  1.0  1.0  1.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0   
  # 19997  0.0  1.0  1.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0   
  # 19998  0.0  1.0  1.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0   
  # 19999  0.0  1.0  1.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0   
  # 20000  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0   

  #       V90  V91  V92  V93  V94  V95  V96  V97  V98  V99  V100  V101  V102  \
  # 0      0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0   0.0   0.0   0.0   
  # 1      0.0  0.0  0.0  0.0  0.0  0.0  3.0  1.0  0.0  3.0   1.0   0.0   0.0   
  # 2      1.0  1.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  1.0   0.0   0.0   0.0   
  # 3      0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0   0.0   0.0   0.0   
  # 4      1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0   0.0   0.0   
  # ...    ...  ...  ...  ...  ...  ...  ...  ...  ...  ...   ...   ...   ...   
  # 19996  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0   0.0   0.0   
  # 19997  0.0  0.0  0.0  0.0  0.0  0.0  6.0  6.0  0.0  3.0   3.0   0.0   3.0   
  # 19998  1.0  1.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  1.0   0.0   0.0   0.0   
  # 19999  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0   0.0   0.0   
  # 20000  0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0   0.0   1.0   1.0   

  #       V103  V104  V105  V106  V107  V108  V109  V110  V111  V112  V113  V114  \
  # 0       0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 1       0.0   0.0   0.0   0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 2       0.0   0.0   0.0   0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 3       0.0   0.0   0.0   0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 4       0.0   0.0   0.0   0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # ...     ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 19996   0.0   0.0   0.0   0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 19997   3.0   0.0   0.0   0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 19998   0.0   0.0   0.0   0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 19999   0.0   0.0   0.0   0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 20000   1.0   0.0   0.0   0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   

  #       V115  V116  V117  V118  V119  V120  V121  V122  V123  V124  V125  \
  # 0       1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   2.0   2.0   2.0   
  # 1       1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 2       1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 3       1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 4       1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # ...     ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 19996   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 19997   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 19998   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 19999   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   
  # 20000   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   

  #           V126         V127       V128  V129         V130    V131     V132  \
  # 0      724.0000   724.000000   724.0000   0.0     0.000000     0.0   0.0000   
  # 1        0.0000   112.000000    29.0000   0.0   112.000000    29.0   0.0000   
  # 2        0.0000    47.950001     0.0000   0.0    47.950001     0.0   0.0000   
  # 3        0.0000     0.000000     0.0000   0.0     0.000000     0.0   0.0000   
  # 4        0.0000     0.000000     0.0000   0.0     0.000000     0.0   0.0000   
  # ...         ...          ...        ...   ...          ...     ...      ...   
  # 19996    0.0000     0.000000     0.0000   0.0     0.000000     0.0   0.0000   
  # 19997    0.0000  2242.500000  2242.5000   0.0  1048.000000  1048.0   0.0000   
  # 19998    0.0000   209.949997     0.0000   0.0   209.949997     0.0   0.0000   
  # 19999    0.0000     0.000000     0.0000   0.0     0.000000     0.0   0.0000   
  # 20000   21.7024    21.702400    21.7024   0.0     0.000000     0.0  21.7024   

  #             V133       V134   V135   V136   V137  V279  V280  V281  V282  \
  # 0         0.0000     0.0000  724.0  724.0  724.0   0.0   0.0   0.0   1.0   
  # 1         0.0000     0.0000    0.0    0.0    0.0   0.0   1.0   0.0   0.0   
  # 2         0.0000     0.0000    0.0    0.0    0.0   0.0   0.0   0.0   1.0   
  # 3         0.0000     0.0000    0.0    0.0    0.0   1.0   1.0   0.0   2.0   
  # 4         0.0000     0.0000    0.0    0.0    0.0   0.0   0.0   0.0   0.0   
  # ...          ...        ...    ...    ...    ...   ...   ...   ...   ...   
  # 19996     0.0000     0.0000    0.0    0.0    0.0   0.0   0.0   0.0   0.0   
  # 19997  1194.5000  1194.5000    0.0    0.0    0.0   0.0   9.0   0.0   0.0   
  # 19998     0.0000     0.0000    0.0    0.0    0.0   0.0   0.0   0.0   0.0   
  # 19999     0.0000     0.0000    0.0    0.0    0.0   0.0   0.0   0.0   1.0   
  # 20000    21.7024    21.7024    0.0    0.0    0.0   8.0   8.0   0.0   9.0   

  #       V283  V284  V285  V286  V287  V288  V289  V290  V291  V292  V293  V294  \
  # 0       1.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   1.0   1.0   0.0   0.0   
  # 1       0.0   0.0   3.0   0.0   1.0   0.0   0.0   1.0   1.0   1.0   0.0   0.0   
  # 2       1.0   0.0   1.0   0.0   0.0   1.0   1.0   1.0   1.0   1.0   0.0   0.0   
  # 3       2.0   0.0   0.0   0.0   0.0   0.0   0.0   2.0   2.0   2.0   1.0   1.0   
  # 4       0.0   0.0   1.0   0.0   0.0   0.0   0.0   1.0   2.0   1.0   0.0   0.0   
  # ...     ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 19996   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   1.0   1.0   0.0   0.0   
  # 19997   4.0   0.0   7.0   0.0   6.0   0.0   1.0   1.0   2.0   2.0   0.0   3.0   
  # 19998   0.0   0.0   1.0   0.0   0.0   0.0   0.0   1.0   1.0   1.0   0.0   0.0   
  # 19999   1.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   1.0   1.0   0.0   0.0   
  # 20000   9.0   0.0   0.0   0.0   0.0   0.0   0.0   5.0   5.0   5.0   8.0   8.0   

  #       V295  V296  V297  V298  V299  V300  V301  V302  V303  V304  V305  \
  # 0       0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   
  # 1       0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   
  # 2       0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   
  # 3       1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   
  # 4       0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   
  # ...     ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   
  # 19996   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   
  # 19997   3.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   
  # 19998   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   
  # 19999   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   1.0   1.0   1.0   
  # 20000   8.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   1.0   1.0   1.0   

  #             V306         V307         V308  V309         V310  V311    V312  \
  # 0        0.000000     0.000000     0.000000   0.0     0.000000   0.0     0.0   
  # 1        0.000000   112.000000    29.000000   0.0   112.000000   0.0    29.0   
  # 2        0.000000    47.950001     0.000000   0.0    47.950001   0.0     0.0   
  # 3      100.599297   100.599297   100.599297   0.0     0.000000   0.0     0.0   
  # 4        0.000000   107.949997     0.000000   0.0   107.949997   0.0     0.0   
  # ...           ...          ...          ...   ...          ...   ...     ...   
  # 19996    0.000000     0.000000     0.000000   0.0     0.000000   0.0     0.0   
  # 19997    0.000000  3338.500000  3058.500000   0.0  2144.000000   0.0  1864.0   
  # 19998    0.000000   209.949997     0.000000   0.0   209.949997   0.0     0.0   
  # 19999    0.000000     0.000000     0.000000   0.0     0.000000   0.0     0.0   
  # 20000  195.321701   195.321701   195.321701   0.0     0.000000   0.0     0.0   

  #             V313        V314        V315        V316         V317  \
  # 0       0.000000    0.000000    0.000000    0.000000     0.000000   
  # 1       0.000000    0.000000    0.000000    0.000000     0.000000   
  # 2      87.949997  135.899994   87.949997    0.000000     0.000000   
  # 3       0.000000    0.000000    0.000000  100.599297   100.599297   
  # 4       0.000000    0.000000    0.000000    0.000000     0.000000   
  # ...          ...         ...         ...         ...          ...   
  # 19996   0.000000    0.000000    0.000000    0.000000     0.000000   
  # 19997   0.000000  706.000000  200.000000    0.000000  1194.500000   
  # 19998   0.000000    0.000000    0.000000    0.000000     0.000000   
  # 19999   0.000000    0.000000    0.000000    0.000000     0.000000   
  # 20000   0.000000    0.000000    0.000000  195.321701   195.321701   

  #               V318  V319  V320  V321  
  # 0         0.000000   0.0   0.0   0.0  
  # 1         0.000000   0.0   0.0   0.0  
  # 2         0.000000   0.0   0.0   0.0  
  # 3       100.599297   0.0   0.0   0.0  
  # 4         0.000000   0.0   0.0   0.0  
  # ...            ...   ...   ...   ...  
  # 19996     0.000000   0.0   0.0   0.0  
  # 19997  1194.500000   0.0   0.0   0.0  
  # 19998     0.000000   0.0   0.0   0.0  
  # 19999     0.000000   0.0   0.0   0.0  
  # 20000   195.321701   0.0   0.0   0.0  

  # [20001 rows x 191 columns]


  # numerical_df_MiceImputed=under40_nan_numerical_df.copy(deep=True) 
  mice_imputer=IterativeImputer()
  under40_nan_numerical_df.iloc[:, :]=mice_imputer.fit_transform(under40_nan_numerical_df)
  # print('numerical_df_MiceImputed',numerical_df_MiceImputed)

  number_of_nan_in_entire_columns=under40_nan_numerical_df.isnull().sum(axis=0).sum()
  # print('number_of_nan_in_entire_columns',number_of_nan_in_entire_columns)

  assert number_of_nan_in_entire_columns==0,'number_of_nan_in_entire_columns!=0'

  with open('./pickles/imputed_numerical_df_test.pkl','wb') as f:
    pickle.dump(under40_nan_numerical_df,f)
  
  return under40_nan_numerical_df

def encode_categorical_data_using_LabelEncoder(imputed_categorical_df):
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

  with open('./pickles/imputed_categorical_df.pkl','wb') as f:
    pickle.dump(imputed_categorical_df,f)

  return imputed_categorical_df

def concat_numerical_and_categorical_data(imputed_numerical_df,encoded_categorical_data):
  concated_data=pd.merge(imputed_numerical_df,encoded_categorical_data,on=['TransactionID'],how='left')
  return concated_data

def perform_normalization(train_X):
  # print('train_X.mean()',round(train_X.mean()))
  # print('train_X.std()',train_X.std())
  normalized_train_X=(train_X-train_X.mean())/(train_X.std()+1e-6)
  # print('normalized_df',normalized_df)
  return normalized_train_X

def evaluate_by_lgbm(normalized_train_X):
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

def optimize_hyperparameter_of_lgbm(normalized_train_X,smote_train_y):
  
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
  
  return bestP

def split_train_and_validation_before_oversampling(concated_numerical_categorical_data):
  # print('concated_numerical_categorical_data',concated_numerical_categorical_data)
  
  concated_numerical_categorical_data=shuffle(concated_numerical_categorical_data)

  # ================================================================================
  full_normal=concated_numerical_categorical_data[concated_numerical_categorical_data['isFraud']==0]
  full_fraud=concated_numerical_categorical_data[concated_numerical_categorical_data['isFraud']==1]

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

def impute_numerical_data_by_mean(under40_nan_numerical_df):
  # print('under40_nan_numerical_df',under40_nan_numerical_df)
  
  # under40_nan_numerical_df.fillna(under40_nan_numerical_df.mean(),inplace=True)

  # under40_nan_numerical_df.apply(lambda x:x.fillna(x.mean(),inplace=True))
  # print('imputed',imputed.dtypes)
  # print('imputed',imputed)
  # print('imputed',imputed.isnull().sum(axis=0))

  temp_df=[]
  for one_column_name in under40_nan_numerical_df:
    # print('one_column_name',one_column_name)
    # one_column_name TransactionID

    if one_column_name=="TransactionID":
      temp_df.append(under40_nan_numerical_df[[one_column_name]])
      continue
    
    one_df=under40_nan_numerical_df[[one_column_name]]
    one_df_mean=one_df.mean()
    temp_df.append(one_df.fillna(one_df_mean))
  
  temp_df2=pd.concat(temp_df,axis=1)
  # print('temp_df2',temp_df2)

  # ================================================================================
  number_of_nan_in_entire_columns=temp_df2.isnull().sum(axis=0).sum()
  assert number_of_nan_in_entire_columns==0,'number_of_nan_in_entire_columns!=0'

  with open('./pickles/imputed_numerical_df.pkl','wb') as f:
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
  # imputed_numerical_df_columns.remove("isFraud")
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

  fn='./results_csv/sample_submission.csv'
  merged.to_csv(fn,sep=',',encoding='utf-8',index=False)

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
      # aafaf

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

# ================================================================================
# Load csv file

csv_train=load_csv_files()
# print('csv_train',csv_train)

# ================================================================================
# check NaN in column and row
# Not use this function

# nan_ratio_df=check_nan(csv_train_transaction_original,create_image=CREATE_IMAGE_ON_NAN_RATIO)

# ================================================================================
# Discard columns whose NaNs are too abundant
# Not use this function

# under40_nan_df,categorical_features,numerical_features=discard_40percent_nan_columns(csv_train_transaction_original,nan_ratio_df)
# del csv_train_transaction_original
# gc.collect()

# ================================================================================
# Separate full column data into categorical data and numerical data 

numerical_df,categorical_df=separate_full_column_data_into_categorical_and_numerical(csv_train)
numerical_df=numerical_df.astype("float32")
# print("numerical_df@",numerical_df)
# print("categorical_df@",categorical_df)

del csv_train
gc.collect()

# ================================================================================
# Convert time delta

numerical_df=convert_time_delte(numerical_df)
# print('numerical_df',numerical_df)

# ================================================================================
# Impute null of categorical data by mode

imputed_categorical_df=impute_categorical_data_by_mode(categorical_df)

del categorical_df
gc.collect()

# ================================================================================
imputed_categorical_df=categorical_other(imputed_categorical_df)
# print('imputed_categorical_df',imputed_categorical_df)

# ================================================================================
# Combine feature to create new features

first_new_feature_added=encode_CB('card1','addr1',imputed_categorical_df)
categorical_df=encode_CB('card1_addr1','P_emaildomain',first_new_feature_added)

# ================================================================================
# Impute null of numerical data by MICE

if IMPUTED_NUMERICAL_DATA_SOURCE=="function":
  if METHOD_FOR_IMPUTE_NUMERICAL_DATA=="mice":
    imputed_numerical_df=impute_numerical_data_by_MICE(numerical_df)
  elif METHOD_FOR_IMPUTE_NUMERICAL_DATA=="mean":
    imputed_numerical_df=impute_numerical_data_by_mean(numerical_df)
elif IMPUTED_NUMERICAL_DATA_SOURCE=="pickle":
  with open('./pickles/imputed_numerical_df.pkl','rb') as f:
    imputed_numerical_df=pickle.load(f)

del numerical_df
gc.collect()

# ================================================================================
# Manage skewness

# skewness_managed_numerical_df=manage_skewness(imputed_numerical_df)
# del imputed_numerical_df
# gc.collect()

skewness_managed_numerical_df=imputed_numerical_df
del imputed_numerical_df
gc.collect()

# ================================================================================
# Compress data

# csv_train_identity=reduce_mem_usage(csv_train_identity)
# compressed_imputed_numerical_df=reduce_mem_usage(imputed_numerical_df)

# ================================================================================
# Encode categorical data using LabelEncoder

if ENCODED_CATEGORICAL_DATA_SOURCE=="function":
  encoded_categorical_data=encode_categorical_data_using_LabelEncoder(imputed_categorical_df)
elif ENCODED_CATEGORICAL_DATA_SOURCE=="pickle":
  with open('./pickles/imputed_categorical_df.pkl','rb') as f:
    encoded_categorical_data=pickle.load(f)

del imputed_categorical_df
gc.collect()

# ================================================================================
# Concatenate numerical and categorical data

concated_numerical_categorical_data=concat_numerical_and_categorical_data(skewness_managed_numerical_df.astype("float32"),encoded_categorical_data)
# print('concated_numerical_categorical_data',concated_numerical_categorical_data)

del skewness_managed_numerical_df
del encoded_categorical_data
gc.collect()

# ================================================================================
# Perform normalization on all columns

# print('smote_train_X',smote_train_X)
# normalized_train_X=perform_normalization(concated_numerical_categorical_data)
# print('normalized_train_X',normalized_train_X)

# del concated_numerical_categorical_data
# gc.collect()

# ================================================================================
# print('concated_numerical_categorical_data',concated_numerical_categorical_data)
predictions=evaluate_by_lgbm(concated_numerical_categorical_data)

# ================================================================================
# Create submission

create_submission(predictions)
