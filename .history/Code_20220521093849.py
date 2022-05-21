#%%
# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

pd.set_option('display.max_columns', 36)
pd.set_option('display.max_row', 36)

#%%%
# Data set này mô tả chứa 119390 observations về City Hotel and Resort Hotel 
# Mỗi observation đại diện cho một hotel booking, bao gồm các đặt phòng đã đến và bị hủy
df = pd.read_csv('hotel_bookings.csv')
print('\n____________ Dataset info ____________')
print(df.info())
print('\n____________ Some first data examples ____________')
print(df.head(3))
print('\n____________ Statistics of numeric features ____________')
print(df.describe())
# checking for null values 
print('\n____________ Null Values ____________')
null = pd.DataFrame({'Null Values' : df.isna().sum(), 'Percentage Null Values' : (df.isna().sum()) / (df.shape[0]) * (100)})
null
# filling null values with zero
print('\n____________ Filling Null Values With Zero ____________')
df.fillna(0, inplace = True)
# %%


