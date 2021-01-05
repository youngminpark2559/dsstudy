# conda activate py36_django_bare && \
# cd /mnt/external_disk/Companies/side_project/Kaggle/00001_ieee-fraud-detection/public_notebooks/00015_roguLINA_Kaggle_fraud && \
# rm e.l && python fraud-detection-2.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================

# X_train, X_val, y_train, y_val = train_test_split(df.drop(target, axis=1), 
#                                                   df[target], 
#                                                   test_size=0.33, 
#                                                   shuffle=True, 
#                                                   stratify=df[target])

# ================================================================================
import traceback
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time,timeit,datetime
from scipy.stats import rankdata
pd.set_option('display.max_colwidth',None)
pd.set_option('display.max_columns',None)

# ================================================================================
NAN_CRITERION=0.5
# NAN_CRITERION=0.9

# ================================================================================
# Full train Full test
# train_transaction_path="/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/train_transaction_original.csv"
# test_transaction_path="/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/test_transaction_original.csv"

# Full train Small test
train_transaction_path="/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/train_transaction_original.csv"
test_transaction_path="/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/test_transaction.csv"

# Small train Full test
# train_transaction_path="/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/train_transaction_small.csv"
# test_transaction_path="/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/test_transaction_original.csv"

# Small train Small test
# train_transaction_path="/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/train_transaction_small.csv"
# test_transaction_path="/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/test_transaction.csv"

train_identity_path="/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/train_identity.csv"
test_identity_path="/mnt/1T-5e7/Data_collection/Kaggle_00001_ieee-fraud-detection/test_identity.csv"

# ================================================================================
train_identity = pd.read_csv(train_identity_path)

# ================================================================================
# for col in train_identity.columns:
#     if type(train_identity[col].iloc[0]) == np.float64:
#         train_identity[col] = train_identity[col].astype(np.float16)
# #train_identity.to_csv('/kaggle/low_dim/train_transaction.csv')

# ================================================================================
train_transaction = pd.read_csv(train_transaction_path)

# ================================================================================
# for col in train_transaction.columns:
#     if type(train_transaction[col].iloc[0]) == np.float64:
#         train_transaction[col] = train_transaction[col].astype(np.float16)
# train_transaction.to_csv('/kaggle/low_dim/train_transaction.csv')

# ================================================================================
target = 'isFraud'

# ================================================================================
# df = pd.merge(train_identity, train_transaction, on='TransactionID', how='outer')
df = pd.merge(train_identity, train_transaction, on='TransactionID', how='right')

# ================================================================================
# drop columns where more than 50% of NaNs

drop_cols = df.isna().sum()[df.isna().sum() > (df.shape[0] * NAN_CRITERION)].index.tolist()
# print("drop_cols",drop_cols)
# drop_cols ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'dist1', 'dist2', 'R_emaildomain', 'D5', 'D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14', 'M5', 'M7', 'M8', 'M9', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181', 'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253', 'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278', 'V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339']

# ================================================================================
# df columns after drop the columns 

df.drop(drop_cols, axis=1, inplace=True)
# print('df',list(df.columns))
# ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D10', 'D11', 'D15', 'M1', 'M2', 'M3', 'M4', 'M6', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321']

# ================================================================================
cat_cols_transaction = ['ProductCD',  
                        'addr1', 
                        'addr2',
                        'P_emaildomain',
                        'R_emaildomain']
cat_cols_transaction.extend(['card{}'.format(i) for i in range(1, 7)])
cat_cols_transaction.extend(['M{}'.format(i) for i in range(1, 10)])

cat_cols_identity = ['DeviceType',
                     'DeviceInfo']
cat_cols_identity.extend(['id_{}'.format(i) for i in range(12, 39)])

# print('categorical_columns_transaction: ', cat_cols_transaction)
# print('categorical_columns_identity: ', cat_cols_identity)
# categorical_columns_transaction:  ['ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
# categorical_columns_identity:  ['DeviceType', 'DeviceInfo', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']

# ================================================================================
cat_cols_tr = []
for col in cat_cols_transaction:
    if col not in drop_cols:
        cat_cols_tr.append(col)

cat_cols_id = []
for col in cat_cols_identity:
    if col not in drop_cols:
        cat_cols_id.append(col)
        
# print('categorical_columns_transaction: ', cat_cols_tr)
# print('categorical_columns_identity: ', cat_cols_id)
# categorical_columns_transaction:  ['ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'M1', 'M2', 'M3', 'M4', 'M6']
# categorical_columns_identity:  []

# ================================================================================
# for col in cat_cols_id:
#     if df[col].nunique() < 3:
#         print(col, df[col].unique())

for col in cat_cols_tr:
    if df[col].nunique() < 3:
        print(col, df[col].unique())

# From categorical columns transaction, columns which have the number of unique values less than 3
# M1 [nan 'T' 'F']
# M2 [nan 'T' 'F']
# M3 [nan 'T' 'F']
# M6 [nan 'T' 'F']

# ================================================================================
# train_identity[cat_cols_id].fillna(-1, inplace=True)
df[cat_cols_tr].fillna(-1, inplace=True)

# ================================================================================
# not_cat_cols_id = [col for col in df.columns if col not in cat_cols_id and col != target]
not_cat_cols_tr = [col for col in df.columns if col not in cat_cols_tr and col != target]
# print('numerical_columns_transaction',not_cat_cols_tr)
# numerical_columns_transaction
# ['TransactionID', 'TransactionDT', 'TransactionAmt', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D10', 'D11', 'D15', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321']

# ================================================================================
# df[not_cat_cols_id].apply(lambda x: x.fillna(x.values.mean(), inplace=True))

try:
    df[not_cat_cols_tr].apply(lambda x: x.fillna(x.values.mean(), inplace=True))
except:
    print(traceback.format_exc())

# ================================================================================
try:
    df['V317'].unique()
    # array([   0.        ,  100.        ,   48.63809967, ..., 3816.        ,
    #         680.5       , 1908.        ])
except:
  print(traceback.format_exc())

# ================================================================================
# Sort by TransactionID
df=df.sort_values(by=['TransactionID'],axis=0)

# ================================================================================
# TransactionDT - timedelta. Suggest that it is the number of seconds since 1st January 2017

# ================================================================================
df['timestamp'] = pd.to_datetime(df['TransactionDT'], unit='s')

# ================================================================================
# for x in df['timestamp']:
#     if x.year != 1970:
#         print(x)

# All rows are belonged to one year

# ================================================================================
# Assign consecutive numbers to each datetime 

# print("df['timestamp']",df['timestamp'])
# 0        1970-01-02 00:01:46
# 1        1970-01-02 00:02:15
# 2        1970-01-02 00:02:29
# 3        1970-01-02 00:02:35
# 4        1970-01-02 00:03:40
#                  ...        
# 590535   1970-07-02 23:57:27
# 590536   1970-07-02 23:57:29
# 590537   1970-07-02 23:57:59
# 590538   1970-07-02 23:58:08
# 590539   1970-07-02 23:58:51
# Name: timestamp, Length: 590540, dtype: datetime64[ns]

start_date = df['timestamp'].iloc[0].date()

# ================================================================================
# start=timeit.default_timer()

# ================================================================================
# Loop

# i = 0 
# j = 0
# date_month_agg = []
# while i < len(df):
#     #print(df['timestamp'].iloc[i].date(), start_date, df['timestamp'].iloc[i].date() == start_date)
#     if df['timestamp'].iloc[i].date() == start_date:
#         # start date --> 0
#         date_month_agg.append(j)
#     else:
#         j += 1
#         # Next time
#         start_date = df['timestamp'].iloc[i].date()
#         # Next time -> j+1
#         date_month_agg.append(j)
#     i += 1
# len(date_month_agg) == len(df)
# # print('len(date_month_agg)',len(date_month_agg))
# # print('len(df)',len(df))
# # len(date_month_agg) 590540
# # len(df) 590540

# df['date_month_agg'] = date_month_agg

# # with pd.option_context('display.max_rows',100000):
# #   print(df[['timestamp', 'date_month_agg']])

# fn='./temp_loop.csv'
# df[['timestamp', 'date_month_agg']].to_csv(fn,sep=',',encoding='utf-8',index=False)

# df['date_month_agg'] = date_month_agg
# df[['timestamp', 'date_month_agg']]

# ================================================================================
# Rankdata 

temp_date=df['timestamp'].dt.strftime('%Y-%m-%d')
# print('temp_date',temp_date)
# 0         1970-01-02
# 1         1970-01-02
# 2         1970-01-02
# 3         1970-01-02
# 4         1970-01-02
#              ...    
# 590535    1970-07-02
# 590536    1970-07-02
# 590537    1970-07-02
# 590538    1970-07-02
# 590539    1970-07-02

aaa=rankdata(list(temp_date),method='dense')
df['date_month_agg'] = aaa

# with pd.option_context('display.max_rows',100000):
#   print(df[['timestamp', 'date_month_agg_by_rankdata']])

# fn='./temp_rankdata.csv'
# df[['timestamp', 'date_month_agg_by_rankdata']].to_csv(fn,sep=',',encoding='utf-8',index=False)

# ================================================================================
# Map 

# ymd_from_timestamp=df['timestamp'].dt.strftime('%Y-%m-%d')

# temp_date=ymd_from_timestamp.unique()
# # print('temp_date',temp_date)
# # ['1970-01-02' '1970-01-03' '1970-01-04' '1970-01-05' '1970-01-06'
# #  '1970-01-07' '1970-01-08' '1970-01-09' '1970-01-10' '1970-01-11'

# mapping_table_dict={}
# for idx,element in enumerate(temp_date):
#     mapping_table_dict[element]=idx+1
# # print('mapping_table_dict',mapping_table_dict)
# # mapping_table_dict {'1970-01-02': 1, '1970-01-03': 2, '1970-01-04': 3, '1970-01-05': 4, '1970-01-06': 5, '1970-01-07': 6, '1970-01-08': 7, '1970-01-09': 8, '1970-01-10': 9, '1970-01-11': 10, '1970-01-12': 11, '1970-01-13': 12, '1970-01-14': 13, '1970-01-15': 14, '1970-01-16': 15, '1970-01-17': 16, '1970-01-18': 17, '1970-01-19': 18, '1970-01-20': 19, '1970-01-21': 20, '1970-01-22': 21, '1970-01-23': 22, '1970-01-24': 23, '1970-01-25': 24, '1970-01-26': 25, '1970-01-27': 26, '1970-01-28': 27, '1970-01-29': 28, '1970-01-30': 29, '1970-01-31': 30, '1970-02-01': 31, '1970-02-02': 32, '1970-02-03': 33, '1970-02-04': 34, '1970-02-05': 35, '1970-02-06': 36, '1970-02-07': 37, '1970-02-08': 38, '1970-02-09': 39, '1970-02-10': 40, '1970-02-11': 41, '1970-02-12': 42, '1970-02-13': 43, '1970-02-14': 44, '1970-02-15': 45, '1970-02-16': 46, '1970-02-17': 47, '1970-02-18': 48, '1970-02-19': 49, '1970-02-20': 50, '1970-02-21': 51, '1970-02-22': 52, '1970-02-23': 53, '1970-02-24': 54, '1970-02-25': 55, '1970-02-26': 56, '1970-02-27': 57, '1970-02-28': 58, '1970-03-01': 59, '1970-03-02': 60, '1970-03-03': 61, '1970-03-04': 62, '1970-03-05': 63, '1970-03-06': 64, '1970-03-07': 65, '1970-03-08': 66, '1970-03-09': 67, '1970-03-10': 68, '1970-03-11': 69, '1970-03-12': 70, '1970-03-13': 71, '1970-03-14': 72, '1970-03-15': 73, '1970-03-16': 74, '1970-03-17': 75, '1970-03-18': 76, '1970-03-19': 77, '1970-03-20': 78, '1970-03-21': 79, '1970-03-22': 80, '1970-03-23': 81, '1970-03-24': 82, '1970-03-25': 83, '1970-03-26': 84, '1970-03-27': 85, '1970-03-28': 86, '1970-03-29': 87, '1970-03-30': 88, '1970-03-31': 89, '1970-04-01': 90, '1970-04-02': 91, '1970-04-03': 92, '1970-04-04': 93, '1970-04-05': 94, '1970-04-06': 95, '1970-04-07': 96, '1970-04-08': 97, '1970-04-09': 98, '1970-04-10': 99, '1970-04-11': 100, '1970-04-12': 101, '1970-04-13': 102, '1970-04-14': 103, '1970-04-15': 104, '1970-04-16': 105, '1970-04-17': 106, '1970-04-18': 107, '1970-04-19': 108, '1970-04-20': 109, '1970-04-21': 110, '1970-04-22': 111, '1970-04-23': 112, '1970-04-24': 113, '1970-04-25': 114, '1970-04-26': 115, '1970-04-27': 116, '1970-04-28': 117, '1970-04-29': 118, '1970-04-30': 119, '1970-05-01': 120, '1970-05-02': 121, '1970-05-03': 122, '1970-05-04': 123, '1970-05-05': 124, '1970-05-06': 125, '1970-05-07': 126, '1970-05-08': 127, '1970-05-09': 128, '1970-05-10': 129, '1970-05-11': 130, '1970-05-12': 131, '1970-05-13': 132, '1970-05-14': 133, '1970-05-15': 134, '1970-05-16': 135, '1970-05-17': 136, '1970-05-18': 137, '1970-05-19': 138, '1970-05-20': 139, '1970-05-21': 140, '1970-05-22': 141, '1970-05-23': 142, '1970-05-24': 143, '1970-05-25': 144, '1970-05-26': 145, '1970-05-27': 146, '1970-05-28': 147, '1970-05-29': 148, '1970-05-30': 149, '1970-05-31': 150, '1970-06-01': 151, '1970-06-02': 152, '1970-06-03': 153, '1970-06-04': 154, '1970-06-05': 155, '1970-06-06': 156, '1970-06-07': 157, '1970-06-08': 158, '1970-06-09': 159, '1970-06-10': 160, '1970-06-11': 161, '1970-06-12': 162, '1970-06-13': 163, '1970-06-14': 164, '1970-06-15': 165, '1970-06-16': 166, '1970-06-17': 167, '1970-06-18': 168, '1970-06-19': 169, '1970-06-20': 170, '1970-06-21': 171, '1970-06-22': 172, '1970-06-23': 173, '1970-06-24': 174, '1970-06-25': 175, '1970-06-26': 176, '1970-06-27': 177, '1970-06-28': 178, '1970-06-29': 179, '1970-06-30': 180, '1970-07-01': 181, '1970-07-02': 182}

# df['date_month_agg_by_apply']=ymd_from_timestamp.map(mapping_table_dict)

# fn='./temp_apply.csv'
# df[['timestamp', 'date_month_agg_by_apply']].to_csv(fn,sep=',',encoding='utf-8',index=False)

# ================================================================================
# stop=timeit.default_timer()
# took_time_sec=stop-start
# took_time_min=str(datetime.timedelta(seconds=took_time_sec))
# print('took_time_min',took_time_min)
# loop : took_time_min 0:00:15.650694
# rankdata : took_time_min 0:00:03.045115
# map : took_time_min 0:00:03.034748

# ================================================================================
# Encoding of categorical features

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df[cat_cols_tr] = df[cat_cols_tr].apply(lambda x: le.fit_transform(x.astype(str)))

# ================================================================================
df.drop('timestamp', axis=1, inplace=True)

# ================================================================================
from sklearn.model_selection import train_test_split

# ================================================================================
# X_train, X_val, y_train, y_val = train_test_split(df.drop(target, axis=1), 
#                                                   df[target], 
#                                                   test_size=0.33, 
#                                                   shuffle=True, 
#                                                   stratify=df[target])
# afaf 

# ================================================================================
# from lightgbm import LGBMClassifier
# from sklearn.metrics import roc_auc_score

# clf = LGBMClassifier()
# clf.fit(X_train.values, y_train.values)
# proba = clf.predict_proba(X_val.values)[:, 1]

# print(roc_auc_score(y_val, proba))
# # 0.9261495544920444

# ================================================================================
df_target=df[[target]]
del df[target]

# ================================================================================
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for i,(train_index, test_index) in enumerate(tscv.split(df)):
    # print('Fold : ',i+1)
    # print("TRAIN:", train_index, "TEST:", test_index)
    # print("")

    X_train, X_val = df.iloc[train_index], df.iloc[test_index]
    y_train, y_val = df_target.iloc[train_index], df_target.iloc[test_index]

    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    clf = LGBMClassifier()
    clf.fit(X_train.values, y_train.values)
    proba = clf.predict_proba(X_val.values)[:, 1]

    print(roc_auc_score(y_val, proba))
    # 0.8810430389055413
    # 0.90271522051746
    # 0.8879378600233834
    # 0.9022634461130423
    # 0.9040870062669306

# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================
# # Make a prediction for test dataset

test_identity = pd.read_csv(test_identity_path, names=train_identity.columns)
test_identity.drop(0, axis=0, inplace=True)

# ================================================================================
# for col in test_identity.columns:
#     if type(test_identity[col].iloc[0]) == np.float64:
#         test_identity[col] = test_identity[col].astype(np.float16)
# test_identity.to_csv('/kaggle/low_dim/test_identity.csv')

# ================================================================================
tr_cols = list(train_transaction.columns)
tr_cols.remove(target)
#tr_cols

# ================================================================================
test_transaction = pd.read_csv(test_transaction_path, names=tr_cols)
test_transaction.drop(0, axis=0, inplace=True)
test_transaction.shape

# ================================================================================
# for col in test_transaction.columns:
#     if type(test_transaction[col].iloc[0]) == np.float64:
#         test_transaction[col] = test_transaction[col].astype(np.float16)
# test_transaction.to_csv('/kaggle/low_dim/test_transaction.csv')

# ================================================================================
test = pd.merge(test_identity, test_transaction, on='TransactionID', how='right')

# ================================================================================
test.drop(drop_cols, axis=1, inplace=True)

# ================================================================================
test[cat_cols_tr].fillna(-1, inplace=True)
test[not_cat_cols_tr].apply(lambda x: x.fillna(x.values.astype(np.float32).mean(), inplace=True))

# ================================================================================
test['timestamp'] = pd.to_datetime(test['TransactionDT'], unit='s')
test[['TransactionDT', 'timestamp']]

# ================================================================================
start_date = '1970-07-03'
i = 0 
j = 364
date_month_agg = []
while i < len(test):
    #print(df['timestamp'].iloc[i].date(), start_date, df['timestamp'].iloc[i].date() == start_date)
    if test['timestamp'].iloc[i].date() == start_date:
        date_month_agg.append(j)
    else:
        j += 1
        start_date = test['timestamp'].iloc[i].date()
        date_month_agg.append(j)
    i += 1
len(date_month_agg) == len(test)

# ================================================================================


test['date_month_agg'] = date_month_agg
test[['timestamp', 'date_month_agg']]




test[cat_cols_tr] = test[cat_cols_tr].apply(lambda x: le.fit_transform(x.astype(str)))




test.drop('timestamp', axis=1, inplace=True)




test.shape




clf = LGBMClassifier()
clf.fit(df.drop(target, axis=1), df[target])
proba_test = clf.predict_proba(test.values)[:, 1]
ans_df = pd.DataFrame(columns=['TransactionID', 'isFraud'])
ans_df['TransactionID'] = test['TransactionID']
ans_df['isFraud'] = proba_test
ans_df




# ans_df.drop_duplicates('TransactionID', keep='first', inplace=True)
# ans_df




ans_df.to_csv('/home/other/kaggle/submission_2.csv', index=False)




# !kaggle competitions submit -c ieee-fraud-detection -f submission.csv -m "submission_1"






