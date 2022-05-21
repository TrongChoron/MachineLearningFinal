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
#%%
# People from all over the world are staying in these two hotels. Most guests are from Portugal and other countries in Europe
print('\n____________ How much do guests pay for a room per night? ____________')
df.head()
# Both hotels have different room types and different meal arrangements.Seasonal factors are also important, So the prices varies a lot.
data = df[df['is_canceled'] == 0]

px.box(data_frame = data, x = 'reserved_room_type', y = 'adr', color = 'hotel', template = 'plotly_dark')
# The figure shows that the average price per room depends on its type and the standard deviation.
print('\n____________ How does the price vary per night over the year? ____________')
data_resort = df[(df['hotel'] == 'Resort Hotel') & (df['is_canceled'] == 0)]
data_city = df[(df['hotel'] == 'City Hotel') & (df['is_canceled'] == 0)]
resort_hotel = data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
resort_hotel
city_hotel=data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel
final_hotel = resort_hotel.merge(city_hotel, on = 'arrival_date_month')
final_hotel.columns = ['month', 'price_for_resort', 'price_for_city_hotel']
final_hotel
# So, first we have to provide right hierarchy to month column.
import sort_dataframeby_monthorweek as sd

def sort_month(df, column_name):
    return sd.Sort_Dataframeby_Month(df, column_name)

final_prices = sort_month(final_hotel, 'month')
final_prices

plt.figure(figsize = (17, 8))

px.line(final_prices, x = 'month', y = ['price_for_resort','price_for_city_hotel'],
        title = 'Room price per night over the Months', template = 'plotly_dark')

#%%
# Biểu đồ này cho thấy rõ ràng rằng giá cả ở Khách sạn Resort cao hơn nhiều vào mùa hè và giá của khách sạn thành phố thay đổi ít hơn và đắt nhất là vào mùa Xuân và Thu.

print('\n____________ Which are the most busy months? ____________')
resort_guests = data_resort['arrival_date_month'].value_counts().reset_index()
resort_guests.columns=['month','no of guests']
resort_guests

city_guests = data_city['arrival_date_month'].value_counts().reset_index()
city_guests.columns=['month','no of guests']
city_guests

final_guests = resort_guests.merge(city_guests,on='month')
final_guests.columns=['month','no of guests in resort','no of guest in city hotel']
final_guests

final_guests = sort_month(final_guests,'month')
final_guests

px.line(final_guests, x = 'month', y = ['no of guests in resort','no of guest in city hotel'],
        title='Total no of guests per Months', template = 'plotly_dark')

# City Hotel có nhiều khách hơn vào mùa xuân và mùa thu, khi giá cả cũng cao nhất, vào tháng 7 và tháng 8 thì ít khách hơn, mặc dù giá có thấp hơn.
# Số lượng khách cho Resort Hotel giảm nhẹ từ tháng 6 đến tháng 9, cũng là lúc giá cao nhất. Cả hai khách sạn đều có ít khách nhất trong mùa đông.

# %%
print('\n____________ How long do people stay at the hotels? ____________')
filter = df['is_canceled'] == 0
data = df[filter]
data.head()

data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
data.head()

stay = data.groupby(['total_nights', 'hotel']).agg('count').reset_index()
stay = stay.iloc[:, :3]
stay = stay.rename(columns={'is_canceled':'Number of stays'})
stay

px.bar(data_frame = stay, x = 'total_nights', y = 'Number of stays', color = 'hotel', barmode = 'group',
        template = 'plotly_dark')

#%%

print('\n____________ Data Pre Processing ____________')
plt.figure(figsize = (24, 12))

corr = df.corr()
sns.heatmap(corr, annot = True, linewidths = 1)
plt.show()

correlation = df.corr()['is_canceled'].abs().sort_values(ascending = False)
correlation

# dropping columns that are not useful

useless_col = ['days_in_waiting_list', 'arrival_date_year', 'arrival_date_year', 'assigned_room_type', 'booking_changes',
               'reservation_status', 'country', 'days_in_waiting_list']

df.drop(useless_col, axis = 1, inplace = True)

df.head()

# creating numerical and categorical dataframes

cat_cols = [col for col in df.columns if df[col].dtype == 'O']
cat_cols

cat_df = df[cat_cols]
cat_df.head()

cat_df['reservation_status_date'] = pd.to_datetime(cat_df['reservation_status_date'])

cat_df['year'] = cat_df['reservation_status_date'].dt.year
cat_df['month'] = cat_df['reservation_status_date'].dt.month
cat_df['day'] = cat_df['reservation_status_date'].dt.day

cat_df.drop(['reservation_status_date','arrival_date_month'] , axis = 1, inplace = True)
cat_df.head()

# printing unique values of each column
for col in cat_df.columns:
    print(f"{col}: \n{cat_df[col].unique()}\n")

# Đưa  các biến về các giá trị số để xử lý 

cat_df['hotel'] = cat_df['hotel'].map({'Resort Hotel' : 0, 'City Hotel' : 1})

cat_df['meal'] = cat_df['meal'].map({'BB' : 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4})

cat_df['market_segment'] = cat_df['market_segment'].map({'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3,
                                                           'Complementary': 4, 'Groups': 5, 'Undefined': 6, 'Aviation': 7})

cat_df['distribution_channel'] = cat_df['distribution_channel'].map({'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'Undefined': 3,
                                                                       'GDS': 4})

cat_df['reserved_room_type'] = cat_df['reserved_room_type'].map({'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6,
                                                                   'L': 7, 'B': 8})

cat_df['deposit_type'] = cat_df['deposit_type'].map({'No Deposit': 0, 'Refundable': 1, 'Non Refund': 3})

cat_df['customer_type'] = cat_df['customer_type'].map({'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3})

cat_df['year'] = cat_df['year'].map({2015: 0, 2014: 1, 2016: 2, 2017: 3})

cat_df.head()

num_df = df.drop(columns = cat_cols, axis = 1)
num_df.drop('is_canceled', axis = 1, inplace = True)
num_df

# normalizing numerical variables

num_df['lead_time'] = np.log(num_df['lead_time'] + 1)
num_df['arrival_date_week_number'] = np.log(num_df['arrival_date_week_number'] + 1)
num_df['arrival_date_day_of_month'] = np.log(num_df['arrival_date_day_of_month'] + 1)
num_df['agent'] = np.log(num_df['agent'] + 1)
num_df['company'] = np.log(num_df['company'] + 1)
num_df['adr'] = np.log(num_df['adr'] + 1)

num_df.var()

num_df['adr'] = num_df['adr'].fillna(value = num_df['adr'].mean())

num_df['adr'] = num_df['adr'].fillna(value = num_df['adr'].mean())
num_df.head()

X = pd.concat([cat_df, num_df], axis = 1)
y = df['is_canceled']

X.shape, y.shape

# splitting data into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
X_train.head()
X_test.head()
y_train.head(), y_test.head()

#%%
print('\n____________ Model Building ____________')
print('\n____________ Logistic Regression ____________')

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
conf = confusion_matrix(y_test, y_pred_lr)
clf_report = classification_report(y_test, y_pred_lr)

print(f"Accuracy Score of Logistic Regression is : {acc_lr}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")