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

#%%%
raw_data = pd.read_csv('hotel_booking.csv')
print('\n____________ Dataset info ____________')
print(raw_data.info())
print('\n____________ Dataset info ____________')
print(raw_data.info())
print('\n____________ Some first data examples ____________')
print(raw_data.head(3))
print('\n____________ Statistics of numeric features ____________')
print(raw_data.describe())
# %%
