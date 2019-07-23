
# coding: utf-8

# ## Name : Abdelrahman Yousef Adnan
# ## Date : 23/7/2019
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("bank-full.csv",sep = ';')
df_copy = df.copy()


# In[3]:


df.head(4)


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


print(df['job'].unique())
print(df['marital'].unique())
print(df['education'].unique())
print(df['default'].unique())
print(df['housing'].unique())
print(df['loan'].unique())
print(df['contact'].unique())
print(df['month'].unique())
print(df['poutcome'].unique())
print(df['y'].unique())


# # Encoding

# In[7]:


from sklearn.preprocessing import LabelEncoder


# In[8]:


obj_cols=[]
for i in df.columns:
    if df[i].dtype == "object":
        obj_cols.append(i)
print(obj_cols)


# In[9]:


job_labels= LabelEncoder()
marital_labels = LabelEncoder()
education_labels = LabelEncoder()
default_labels = LabelEncoder()
housing_labels = LabelEncoder()
loan_labels = LabelEncoder()
contact_labels = LabelEncoder()
month_labels = LabelEncoder()
poutcome_labels = LabelEncoder()
y_labels = LabelEncoder()


# In[10]:


df['job']       =job_labels.fit_transform(df['job'])
df['marital']   =marital_labels.fit_transform(df['marital'])
df['education'] =education_labels.fit_transform(df['education'])
df['default']   =default_labels.fit_transform(df['default'])
df['housing']   =housing_labels.fit_transform(df['housing'])
df['loan']      =loan_labels.fit_transform(df['loan'])
df['contact']   =contact_labels.fit_transform(df['contact'])
df['month']     =month_labels.fit_transform(df['month'])
df['poutcome']  =poutcome_labels.fit_transform(df['poutcome'])
df['y']         =y_labels.fit_transform(df['y'])


# In[11]:


df.head()


# In[12]:


df_copy.head()


# In[13]:


from sklearn.preprocessing import StandardScaler


# In[14]:


plt.figure(figsize=(18,8))
sns.heatmap(df.corr(),annot=True);


# In[15]:



# Splitting the dataset into the Training set and Test set
X = df.iloc[:, :-1].values
y = df.iloc[:, 16].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[16]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[20]:


X_train


# In[18]:


y


# In[19]:



# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[21]:



# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[22]:



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[23]:


cm


# In[25]:


accuracy_score(y_test, y_pred)


# In[46]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 30, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[47]:



# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[48]:



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[49]:


cm


# In[50]:


accuracy_score(y_test, y_pred)


# In[52]:



# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())


# In[53]:



# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [10, 20, 30, 50,70,100,150], 'criterion': ['entropy']}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# In[54]:


print(best_accuracy)
print(best_parameters)

