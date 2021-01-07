# conda activate py36_django_bare && \
# cd /home/young && \
# rm e.l && python temppy.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import numpy as np

# ================================================================================
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.datasets import make_blobs
import numpy as np

# --------------------------------------------------------------------------------
# Create example data

X,y=make_blobs(n_samples=12,random_state=0)
# print('X\n',X)
# print('y\n',y)
# X
# [[ 3.54934659  0.6925054 ]
#  [ 1.9263585   4.15243012]
#  [ 0.0058752   4.38724103]
#  [ 1.12031365  5.75806083]
#  [ 1.7373078   4.42546234]
#  [ 2.36833522  0.04356792]
#  [-0.49772229  1.55128226]
#  [-1.4811455   2.73069841]
#  [ 0.87305123  4.71438583]
#  [-0.66246781  2.17571724]
#  [ 0.74285061  1.46351659]
#  [ 2.49913075  1.23133799]]
# y
# [1 0 2 0 0 1 1 2 0 2 2 1]

# --------------------------------------------------------------------------------
# Suppose example data is collected from following group-distribution

groups=[0,0,0,1,1,1,1,2,2,3,3,3]

# --------------------------------------------------------------------------------
# Configure model

leave_one_group_out=LeaveOneGroupOut()
# leave_one_group_out=LeaveOneGroupOut(n_splits=3)
#     leave_one_group_out=LeaveOneGroupOut(n_splits=3)
# TypeError: object() takes no parameters

leave_one_group_out_get_n_splits=leave_one_group_out.get_n_splits(X, y, groups)
# print('leave_one_group_out_get_n_splits',leave_one_group_out_get_n_splits)
# leave_one_group_out_get_n_splits 4

# --------------------------------------------------------------------------------
for train,test in leave_one_group_out.split(X,y,groups):

  print('train',train.shape)
  print('test',test.shape)

  print('X[test]\n',X[train])
  print('y[test]\n',y[test])
  print('train groups',np.array(groups)[train])
  print('test groups',np.array(groups)[test])
  print('')


# ================================================================================
from sklearn.model_selection import GroupKFold
from sklearn.datasets import make_blobs

# --------------------------------------------------------------------------------
# Create example data

X,y=make_blobs(n_samples=12,random_state=0)
# print('X\n',X)
# print('y\n',y)
# X
# [[ 3.54934659  0.6925054 ]
#  [ 1.9263585   4.15243012]
#  [ 0.0058752   4.38724103]
#  [ 1.12031365  5.75806083]
#  [ 1.7373078   4.42546234]
#  [ 2.36833522  0.04356792]
#  [-0.49772229  1.55128226]
#  [-1.4811455   2.73069841]
#  [ 0.87305123  4.71438583]
#  [-0.66246781  2.17571724]
#  [ 0.74285061  1.46351659]
#  [ 2.49913075  1.23133799]]
# y
# [1 0 2 0 0 1 1 2 0 2 2 1]

# --------------------------------------------------------------------------------
# Suppose example data is collected from following group-distribution

groups=[0,0,0,1,1,1,1,2,2,3,3,3]

# --------------------------------------------------------------------------------
# Configure model

group_kfold=GroupKFold(n_splits=4)
# group_kfold=GroupKFold(n_splits=5)
# ValueError: Cannot have number of splits n_splits=5 greater than the number of groups: 4.

group_kfold_get_n_splits=group_kfold.get_n_splits(X, y, groups)
# print('group_kfold_get_n_splits',group_kfold_get_n_splits)
# 3 

# --------------------------------------------------------------------------------
for train,test in group_kfold.split(X,y,groups):

  print('train',train.shape)
  print('test',test.shape)

  print('X[test]\n',X[train])
  print('y[test]\n',y[test])
  print('train groups',np.array(groups)[train])
  print('test groups',np.array(groups)[test])
  print('')

# ================================================================================
from sklearn.model_selection import LeavePGroupsOut
from sklearn.datasets import make_blobs
import numpy as np

# --------------------------------------------------------------------------------
# Create example data

X,y=make_blobs(n_samples=12,random_state=0)
# print('X\n',X)
# print('y\n',y)
# X
# [[ 3.54934659  0.6925054 ]
#  [ 1.9263585   4.15243012]
#  [ 0.0058752   4.38724103]
#  [ 1.12031365  5.75806083]
#  [ 1.7373078   4.42546234]
#  [ 2.36833522  0.04356792]
#  [-0.49772229  1.55128226]
#  [-1.4811455   2.73069841]
#  [ 0.87305123  4.71438583]
#  [-0.66246781  2.17571724]
#  [ 0.74285061  1.46351659]
#  [ 2.49913075  1.23133799]]
# y
# [1 0 2 0 0 1 1 2 0 2 2 1]

# --------------------------------------------------------------------------------
# Suppose example data is collected from following group-distribution

groups=[0,0,0,1,1,1,1,2,2,3,3,3]

# --------------------------------------------------------------------------------
# Configure model

leave_p_group_out=LeavePGroupsOut(n_groups=2)

leave_p_group_out_get_n_splits=leave_p_group_out.get_n_splits(X, y, groups)
# print('leave_p_group_out_get_n_splits',leave_p_group_out_get_n_splits)
# leave_p_group_out_get_n_splits 6

# --------------------------------------------------------------------------------
for train,test in leave_p_group_out.split(X,y,groups):

  print('train',train.shape)
  print('test',test.shape)

  print('X[test]\n',X[train])
  print('y[test]\n',y[test])
  print('train groups',np.array(groups)[train])
  print('test groups',np.array(groups)[test])
  print('')
