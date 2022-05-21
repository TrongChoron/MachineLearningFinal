#%%
# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import folium
from folium.plugins import HeatMap
import seaborn as sns
import plotly.express as px 

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
# visualizing null values
print('\n____________ Visualizing Null Values ____________')
msno.bar(df)
plt.show()
# adults, babies and children cant be zero at same time, so dropping the rows having all these zero at same time
# Filter tất cả các giá trị không phù hợp
filter = (df.children == 0) & (df.adults == 0) & (df.babies == 0)
df[filter]
df = df[~filter]
df
# %%
print('\n____________ Exploratory Data Analysis (EDA) ____________')
print('\n____________ From where the most guests are coming ? ____________')
country_wise_guests = df[df['is_canceled'] == 0]['country'].value_counts().reset_index()
country_wise_guests.columns = ['country', 'No of guests']
country_wise_guests

basemap = folium.Map()
guests_map = px.choropleth(country_wise_guests, locations = country_wise_guests['country'],
                           color = country_wise_guests['No of guests'], hover_name = country_wise_guests['country'])
guests_map.show()
# People from all over the world are staying in these two hotels. Most guests are from Portugal and other countries in Europe
print('\n____________ How much do guests pay for a room per night? ____________')
df.head()
# %%