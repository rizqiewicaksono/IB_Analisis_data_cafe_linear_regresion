#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[4]:


#Loading in the dataset
dataset = pd.read_csv('Documents\CafeShop.csv')
dataset.head()


# In[29]:


#Create a satistical summary table
dataset.describe()


# In[18]:


dataset.columns


# In[19]:


dataset.info()


# In[20]:


stats_summary = dataset.describe()
stats_summary 


# In[22]:


correlation_table = dataset.corr()
correlation_table


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[31]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


x=dataset['jumlah']
y=dataset['kategori']
plt.scatter(x,y)
plt.title ("Jumlah Produk Terbanyak")


# In[44]:


plt.bar(dataset.head(50)['kategori'], dataset.head(50)['jumlah'])


# In[7]:


import seaborn as sns


# In[10]:


sns.boxplot(x='jumlah',data=dataset)


# In[11]:


sns.distplot(dataset['jumlah'])


# In[12]:


sns.pairplot(data=dataset)


# In[14]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[15]:


X = dataset.iloc[:,1:4]
y = dataset['jumlah']
X_train,X_test, y_train,y_test = train_test_split(X,y)




# In[17]:


lin = LinearRegression()


# In[27]:


import pandas as pd 
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[28]:


sia=SentimentIntensityAnalyzer


# In[3]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import  LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


sns.heatmap(dataset.corr(),annot=True);


# In[33]:


dataset = dataset.dropna(axis=0)


# In[6]:


dataset['symmetry'] = dataset['harga']/dataset['jumlah']
dataset.head()


# In[42]:


dataset = dataset.dropna(axis=0)


# In[7]:


dataset_trans=pd.get_dummies(dataset)
X = dataset_trans.drop(['harga','jumlah'],axis=1)
y=dataset_trans['harga']
features = X.columns


# In[8]:


X_train,X_test,y_train,y_test = train_test_split(X,y)


# In[9]:


models_eval = pd.DataFrame(index=['Null','KNN','MLR'],columns=['RMSE'])


# In[10]:



ypred_null = y_train.mean()


# In[11]:



knn = KNeighborsRegressor(n_neighbors=7)
knn.fit(X_train,y_train)


# In[12]:


y_pred = knn.predict(X_test) 


# In[13]:


rf = RandomForestRegressor(n_estimators=10)
rf.fit(X_train,y_train)
y_pred2=y_pred = rf.predict(X_test)


# In[14]:


lin = LinearRegression()
lin = lin.fit(X_train,y_train)
y_pred3 = lin.predict(X_test)


# In[15]:


lasso = Lasso()
lasso.fit(X_train,y_train)
y_pred4 = lasso.predict(X_test)


# In[26]:


model_eval=pd.DataFrame(index=['KNN','MLR'],columns=['RMSE'])
model_eval.loc['KNN','RMSE']=np.sqrt(mean_squared_error(y_test,y_pred))
model_eval.loc['RF','RMSE'] = np.sqrt(mean_squared_error(y_test,y_pred2))
model_eval.loc['MLR','RMSE'] = np.sqrt(mean_squared_error(y_test,y_pred3))
model_eval.loc['Lasso','RMSE'] = np.sqrt(mean_squared_error(y_test,y_pred4))
model_eval.loc['NULL','RMSE'] = ypred_null
model_eval


# In[17]:



fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(y_test,y_pred,s=1)
ax.plot(y_test,y_test,color='red')


# In[18]:


sns.distplot(y_pred-y_test)


# In[19]:


lin=LinearRegression()
lin.fit(X_train,y_train)
y_pred2 = lin.predict(X_test)
model_eval.loc['MLR','RMSE']=np.sqrt(mean_squared_error(y_pred2,y_test))


# In[20]:


from sklearn.ensemble import ExtraTreesRegressor
model =ExtraTreesRegressor()
model.fit(X,y)


# In[21]:


ExtraTreesRegressor()


# In[22]:


model.feature_importances_.tolist()


# In[23]:


plt.figure(figsize=(20,10))
sns.heatmap(dataset.corr(),annot=True);


# In[ ]:




