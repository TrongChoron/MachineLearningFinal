
'''
The dataset get from Kaggle
https://www.kaggle.com/datasets/mojtaba142/hotel-booking?datasetId=1437463&searchQuery=predict

The following code is mainly from Kaggle
https://www.kaggle.com/code/niteshyadav3103/hotel-booking-prediction-99-5-acc/notebook

'''
#%%
# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
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
df = pd.read_csv('NguyenNhatTam_19110283_Data.csv')
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

# adults, babies and children cant be zero at same time, so dropping the rows having all these zero at same time
# Filter tất cả các giá trị không phù hợp
filter = (df.children == 0) & (df.adults == 0) & (df.babies == 0)
df[filter]
df = df[~filter]
df

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
print('\n____________ Model Building  New ____________')

# precision dc tính theo cong thức F1 = 2 * (precision * recall) / (precision + recall)

def eval_prediction(model, pred, x_train, y_train, x_test, y_test):

    print("Accuracy (Test Set): %.2f" % accuracy_score(y_test, pred))
    print("Precision (Test Set): %.2f" % precision_score(y_test, pred))
    print("Recall (Test Set): %.2f" % recall_score(y_test, pred))
    print("F1-Score (Test Set): %.2f" % f1_score(y_test, pred))
    
    fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label=1) # pos_label: label yang kita anggap positive
    print("AUC: %.2f" % auc(fpr, tpr))

def show_best_hyperparameter(model, hyperparameters):
    for key, value in hyperparameters.items() :
        print('Best '+key+':', model.get_params()[key])

def show_cmatrix(ytest, pred):
    # Creating confusion matrix 
    cm = confusion_matrix(ytest, pred)

    # Putting the matrix a dataframe form  
    cm_df = pd.DataFrame(cm, index=['Actually Not Canceled', 'Actually Canceled'],
                 columns=['Predicted Not Canceled', 'Predicted Canceled'])
    
    # visualizing the confusion matrix
    sns.set(font_scale=1.2)
    plt.figure(figsize=(10,4))
        
    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues",xticklabels=cm_df.columns, yticklabels=cm_df.index, annot_kws={"size": 20})
    plt.title("Confusion Matrix", size=20)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class');


#%%
print('\n____________ Logistic Regression ____________')
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
eval_prediction(lr, y_pred_lr, X_train, y_train, X_test, y_test)

print('Train score: ' + str(lr.score(X_train, y_train))) #accuracy
print('Test score:' + str(lr.score(X_test, y_test))) #accuracy

recall_lr = recall_score(y_test, y_pred_lr)
acc_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
acc_lr_train = lr.score(X_train, y_train)

show_cmatrix(y_test, y_pred_lr)

#%%
print('\n____________ KNeighborsClassifier ____________')
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

eval_prediction(knn, y_pred_lr, X_train, y_train, X_test, y_test)

print('Train score: ' + str(knn.score(X_train, y_train))) #accuracy
print('Test score:' + str(knn.score(X_test, y_test))) #accuracy

show_cmatrix(y_test, y_pred_knn)

recall_knn = recall_score(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
acc_knn_train = knn.score(X_train, y_train)

#%%

print('\n____________ Decision Tree Classifier ____________')
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
y_pred_dt = dt_model.predict(X_test)

eval_prediction(dt_model, y_pred_dt, X_train, y_train, X_test, y_test)


print('Train score: ' + str(dt_model.score(X_train, y_train))) #accuracy
print('Test score:' + str(dt_model.score(X_test, y_test))) #accuracy

recall_dt = recall_score(y_test, y_pred_dt)
acc_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
acc_dt_train = dt_model.score(X_train, y_train)

show_cmatrix(y_test,y_pred_dt)

#%%
print('\n____________ Random Forest Classifier ____________')
rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)
y_pred_rf = rf_model.predict(X_test)

eval_prediction(rf_model, y_pred_rf, X_train, y_train, X_test, y_test)

print('Train score: ' + str(rf_model.score(X_train, y_train))) #accuracy
print('Test score:' + str(rf_model.score(X_test, y_test))) #accuracy

recall_rf = recall_score(y_test, y_pred_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
acc_rf_train = rf_model.score(X_train, y_train)

show_cmatrix(y_test, y_pred_rf)

#%%

print('\n____________ Gradient Boosting Classifier ____________')
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)

eval_prediction(gb, y_pred_gb, X_train, y_train, X_test, y_test)

recall_gb = recall_score(y_test, y_pred_gb)
acc_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
acc_gb_train = gb.score(X_train, y_train)

show_cmatrix(y_test, y_pred_gb)

#%%


evaluation_summary = {
    'Logistic Regression': [acc_lr, recall_lr, precision_lr, f1_lr],
    'KNN':[acc_knn, recall_knn, precision_knn, f1_knn],
    'Decision Tree':[acc_dt, recall_dt, precision_dt, f1_dt],
    'Random Forest':[acc_rf, recall_rf, precision_rf, f1_rf],
    'Gradient Boosting':[acc_gb, recall_gb, precision_gb, f1_gb]
}

eva_sum = pd.DataFrame(data = evaluation_summary, index = ['Accuracy', 'Recall', 'Precision', 'F1 Score'])
eva_sum

# tính toán accuracy của train và test
evaluation_sum_train_test = {
    "Train" : [acc_lr_train, acc_knn_train, acc_dt_train, acc_rf_train, acc_gb_train],
    "Test": [acc_lr, acc_knn, acc_dt, acc_rf, acc_gb]
}

eva_sum_train_test = pd.DataFrame(data = evaluation_sum_train_test, index = ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'Gradient Boosting'])
eva_sum_train_test


