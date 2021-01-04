# conda activate py36_django_bare && \
# cd /mnt/external_disk/Companies/side_project/Kaggle/00001_ieee-fraud-detection/public_notebooks/00015_roguLINA_Kaggle_fraud && \
# rm e.l && python fraud-detection-2.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================


# ================================================================================
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time,timeit,datetime
from scipy.stats import rankdata
pd.set_option('display.max_colwidth',None)
pd.set_option('display.max_columns',None)

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
df = pd.merge(train_identity, train_transaction, on='TransactionID', how='outer')

# ================================================================================
# drop columns where more than 50% of NaNs

drop_cols = df.isna().sum()[df.isna().sum() > (df.shape[0] / 2)].index.tolist()
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
df[not_cat_cols_tr].apply(lambda x: x.fillna(x.values.mean(), inplace=True))

# ================================================================================
df['V317'].unique()
# array([   0.        ,  100.        ,   48.63809967, ..., 3816.        ,
#         680.5       , 1908.        ])

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
start=timeit.default_timer()

# ================================================================================
i = 0 
j = 0
date_month_agg = []
while i < len(df):
    #print(df['timestamp'].iloc[i].date(), start_date, df['timestamp'].iloc[i].date() == start_date)
    if df['timestamp'].iloc[i].date() == start_date:
        # start date --> 0
        date_month_agg.append(j)
    else:
        j += 1
        # Next time
        start_date = df['timestamp'].iloc[i].date()
        # Next time -> j+1
        date_month_agg.append(j)
    i += 1
len(date_month_agg) == len(df)
# print('len(date_month_agg)',len(date_month_agg))
# print('len(df)',len(df))
# len(date_month_agg) 590540
# len(df) 590540

df['date_month_agg'] = date_month_agg

# with pd.option_context('display.max_rows',100000):
#   print(df[['timestamp', 'date_month_agg']])

fn='./temp_loop.csv'
df[['timestamp', 'date_month_agg']].to_csv(fn,sep=',',encoding='utf-8',index=False)

# ================================================================================
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
df['date_month_agg_by_rankdata'] = aaa

# with pd.option_context('display.max_rows',100000):
#   print(df[['timestamp', 'date_month_agg_by_rankdata']])

fn='./temp_rankdata.csv'
df[['timestamp', 'date_month_agg_by_rankdata']].to_csv(fn,sep=',',encoding='utf-8',index=False)

afaf   
# print(aaa)
# print(len(aaa))
# 590540









# ================================================================================
stop=timeit.default_timer()
took_time_sec=stop-start
took_time_min=str(datetime.timedelta(seconds=took_time_sec))
# print('took_time_min',took_time_min)
# took_time_min 0:00:15.650694









# ================================================================================
df['date_month_agg'] = date_month_agg
df[['timestamp', 'date_month_agg']]

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
X_train, X_val, y_train, y_val = train_test_split(df.drop(target, axis=1), 
                                                  df[target], 
                                                  test_size=0.33, 
                                                  shuffle=True, 
                                                  stratify=df[target])

print("X_train",X_train)
print("X_val",X_val)
afaf

# ================================================================================
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

clf = LGBMClassifier()
clf.fit(X_train.values, y_train.values)
proba = clf.predict_proba(X_val.values)[:, 1]

roc_auc_score(y_val, proba)

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






