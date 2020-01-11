#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[ ]:





# In[2]:


columns = ['Age','Workclass','fnlgwt','Education','Education num','Marital Status',
           'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',
           'Hours/Week','Native country','Income']


# In[3]:


train = pd.read_csv('C:/Users/Nischal Rajput/adult-training.csv', names=columns)


# In[4]:


test= pd.read_csv('C:/Users/Nischal Rajput/adult-test.csv', names=columns, skiprows=1)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
train.isnull().sum()


# In[6]:


train.info()
train.head()


# In[7]:


test.info()


# In[8]:


train.replace(' ?', np.nan, inplace=True)


# In[9]:



test.replace(' ?', np.nan, inplace=True)


# In[10]:


train.isnull().sum()
train1= train['Age'].apply(np.sum, axis=0)


# In[11]:


train['Income'] = train['Income'].apply(lambda x: 1 if x==' >50K' else 0)
test['Income'] = test['Income'].apply(lambda x: 1 if x==' >50K.' else 0)


# In[12]:


plt.hist(train['Age'])


# In[13]:


train['Workclass'].fillna(0, inplace=True)


# In[14]:


train.Workclass.value_counts()


# In[15]:


test['Workclass'].fillna(0, inplace = True)


# In[ ]:





# In[16]:


plt.hist(test.Age)


# In[17]:


train.fnlgwt.describe()


# In[18]:


train_income = pd.get_dummies(train['Income'])


# In[19]:


train= pd.DataFrame(train,train_income,index=1)


# In[ ]:


train()


# In[ ]:


x=train.iloc[:,0:14]


# In[ ]:


x


# In[ ]:


Y= train.iloc[:,14:]


# In[ ]:


Y


# In[28]:


from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import metrics


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( x, Y, test_size = 0.3, random_state = 0)


# In[ ]:


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


# In[ ]:


pd.get_dummies(train,prefix=['Native country'])


# In[ ]:


train["Native country"] = train["Native country"].astype('category')
train["Sex"] = train["Sex"].astype('category')
train["Race"] = train["Race"].astype('category')


# In[ ]:


train.info()


# In[ ]:


train["Native country"] = train["Native country"].cat.codes
train["Sex"] = train["Sex"].cat.codes


# In[ ]:


train


# In[ ]:


x=train.iloc[:,0:14]


# In[ ]:


x


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( x, Y, test_size = 0.3, random_state = 100)


# In[ ]:


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)


# In[ ]:


train.Race


# In[36]:


X=train[['Education num','Age','Hours/Week']].values
y= train[['Income']].values
X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.3, random_state=21, stratify=y)
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
predn=clf.predict(X_test)
print('The accuracy of the model is',metrics.accuracy_score(predn,y_test))


# In[22]:


X


# In[ ]:


from sklearn import svm

svc = svm.SVC(kernel='linear')

svc.fit(X_train, y_train)

y_pred=svc.predict(X_test)

print("Test set predictions:\n {}".format(y_pred))
print(svc.score(X_test,y_test))


# In[ ]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
param_grid= {'n_neighbors': np.arange(1,80)}
knn = KNeighborsClassifier()
knn_cv=GridSearchCV(knn, param_grid, cv=5)
y = y.reshape(30718,)
knn_cv.fit(X, y)
print(knn_cv.best_params_)
print(knn_cv.best_score_)
model=KNeighborsClassifier(n_neighbors=78) 
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction,y_test))


# In[53]:


train['Occupation'] = train['Occupation'].map({'?': 0, 'Farming-fishing': 1, 'Tech-support': 2, 
                                                       'Adm-clerical': 3, 'Handlers-cleaners': 4, 'Prof-specialty': 5,
                                                       'Machine-op-inspct': 6, 'Exec-managerial': 7, 
                                                       'Priv-house-serv': 8, 'Craft-repair': 9, 'Sales': 10, 
                                                       'Transport-moving': 11, 'Armed-Forces': 12, 'Other-service': 13, 
                                                       'Protective-serv': 14}).astype(int)


# In[51]:


train.head(n=200)


# In[ ]:




