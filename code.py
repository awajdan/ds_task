#loading all relevant and required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

#load csv data file
df = pd.read_csv("data_excel.csv",delimiter=';')
df.drop('Unnamed: 0', axis=1, inplace=True)

#replacing "?" values with nan
df = df.replace(['?'], np.nan)

df.head()

#%%


#check total nan values in dataset
print(df.isna().sum())
msno.matrix(df)

#%%


#dropping rows with nan values in 4 columns
df = df.dropna(subset=['ID_caller','opened_by','location','category_ID'])
df.reset_index(inplace = True)


#%%


#checking for nan values once more
print(df.isna().sum())
msno.matrix(df)


#%%


#Lavel encoding the features, setting categorical data as the selected type

for col in ['ID','ID_status', 'active', 'count_reassign', 'count_opening',
       'count_updated', 'ID_caller', 'opened_by', 'Created_by',
        'updated_by', 'type_contact', 'location', 'category_ID', 'user_symptom','Support_group','impact']:
    df[col] = df[col].astype('category')
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)


#%%


#dropping irrelavant columns

#df = df[['ID_caller', 'opened_by','user_symptom', 'location', 'category_ID','Support_group','impact']]
df = df[['ID_status', 'active', 'count_reassign', 'count_opening',
       'count_updated', 'ID_caller', 'opened_by',
        'updated_by', 'type_contact', 'location', 'category_ID','Support_group','impact']]

df.head()


#%%


#Check the unique values in remaining columns to get an idea 
nu = df.nunique().reset_index()
nu.columns = ['features', 'uniques']
ax=sns.barplot(x='features', y='uniques', data=nu)
ax.tick_params(axis='x', rotation=90)
print(nu)


#%%


#visually check for any obvious correlation
g = sns.heatmap(df.corr(),annot=True,fmt = ".2f", cmap = "coolwarm")
fig=plt.gcf()
fig.set_size_inches(20,20)
plt.show()


#%%


#split data into features and target
from sklearn.model_selection import train_test_split
y = df[['impact','Support_group']]
y.drop('Support_group', axis=1, inplace=True)
X = df.drop(['impact','Support_group'],axis=1)

#split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)
print("Number transactions X_train dataset: ", X_train.shape) 
print("Number transactions y_train dataset: ", y_train.shape) 
print("Number transactions X_test dataset: ", X_test.shape) 
print("Number transactions y_test dataset: ", y_test.shape)


#%%


#check data distributing, data has class imbalance
sns.countplot(x = 'impact',data = y_train)
plt.show()


#%%


#loading more relevant modules and required classes for machine learning from scikit-learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression 




def multiclass_auc(y_test, y_pred, average="macro"):
    """gets the ROC for the target classes"""
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)
     


#%%


#using SMOTE to remove class imbalance
from imblearn.over_sampling import SMOTE as sm

smo = sm(random_state = 2) 
X_train_res, y_train_res = smo.fit_resample(X_train, y_train.values.ravel()) 

l = list(y_train_res)
sns.countplot(x = l)
plt.show()


#%%


#defining model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced')

#fit model on training data
m = model.fit(X_train_res,y_train_res)

#make predictions on test data
y_pred = m.predict(X_test)

#check performance of defined model
y_score = y_pred.score(X_test, y_test, sample_weight=None)
print(y_score)
print(multiclass_auc(y_test,y_pred))
print(classification_report(y_test, y_pred))
