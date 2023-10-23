#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


c21=pd.read_csv('players_21.csv',low_memory=False)


# In[4]:


c22=pd.read_csv('players_22.csv', low_memory=False)


# In[5]:


object_columns = c21.select_dtypes(include=['object']).columns
c21 = c21.drop(columns=object_columns) # i drop all the columns that contiains object, the reason is, they were of any significant value to the model i am about creating 


# In[6]:


object_columns = c22.select_dtypes(include=['object']).columns
c22 = c22.drop(columns=object_columns) # i drop all the columns that contiains object, they were of any significant value to the model i am about creating


# In[7]:


c21.info()


# In[8]:


c22.info()


# In[9]:


c21.corr()['overall'].sort_values(ascending=False) #this help to see the value of each variable and it effect on the model


# In[10]:


c22.corr()['overall'].sort_values(ascending=False)#this help to see the value of each variable and it effect on the model


# In[11]:


selected_columns_21 = c21.corr()['overall'][c21.corr()['overall'] > 0.45].index.tolist()
selected_columns_21 = c21[selected_columns_21]
selected_columns_21    # i took the variables of which their value is greater than 4.5 because they have huge effect on the model and to make the texting quit simple


# In[12]:


selected_columns_22 = c22.corr()['overall'][c22.corr()['overall'] > 0.45].index.tolist()
selected_columns_22 = c22[selected_columns_22]
selected_columns_22


# In[13]:


from matplotlib import pyplot as plt
#Exploratory data analysis (EDA) df_21
selected_columns_21.hist(bins=50, figsize=(20,15))
plt.show()


# In[14]:


from matplotlib import pyplot as plt
#Exploratory data analysis (EDA) df_22
selected_columns_22.hist(bins=50, figsize=(20,15))
plt.show()


# In[15]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(selected_columns_21)
selected_columns_21 = pd.DataFrame(imputed_data, columns=selected_columns_21.columns)
selected_columns_21


# In[16]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(selected_columns_22)
selected_columns_22 = pd.DataFrame(imputed_data, columns=selected_columns_22.columns)
selected_columns_22     #this help to replace missing value with the mean of the whole data


# In[17]:


y_21=selected_columns_21['overall']      #i splitted the data
x_21=selected_columns_21.drop('overall',axis=1)   


# In[18]:


y_22=selected_columns_22['overall']  #i splitted the data
x_22=selected_columns_22.drop('overall',axis=1)


# In[19]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_21)
scaled_data_21 = scaler.transform(x_21)
x_21=pd.DataFrame(scaled_data_21,columns=x_21.columns)
x_21.head() # the code is to ensure standardization to improve the performance of the machine learning algorithm.  


# In[20]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_22)
scaled_data_22 = scaler.transform(x_22)
x_22=pd.DataFrame(scaled_data_22,columns=x_22.columns)
x_22.head()   # the code is to ensure standardization to improve the performance of the machine learning algorithm.


# In[21]:


from sklearn.model_selection import train_test_split  # this is the library that helps in trainig and texting of the model


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(x_21, y_21, test_size=0.2, random_state=42)


# In[23]:


from sklearn.ensemble import RandomForestRegressor


# In[24]:


rf=RandomForestRegressor(max_depth=8, n_estimators=50)


# In[25]:


rf.fit(X_train, y_train)


# In[26]:


ypred=rf.predict(X_test)


# In[27]:


from sklearn.metrics import mean_absolute_error,mean_squared_error  #it is used to check the perfprmance of the model


# In[28]:


mean_absolute_error(y_test, ypred)


# In[29]:


np.sqrt(mean_squared_error(y_test, ypred))


# In[30]:


ypred


# In[31]:


mae = mean_absolute_error(y_test,ypred)

mae


# In[32]:


y_test


# In[33]:


rf.score(X_test,y_test)


# In[34]:


rmse = np.sqrt(mean_squared_error(y_test,ypred))
rmse


# In[35]:


from sklearn.ensemble import GradientBoostingRegressor  #it is used to check the perfprmance of the model


# In[36]:


clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.001, max_depth=3)


# In[37]:


clf.fit(X_train,y_train)


# In[38]:


ypred=clf.predict(X_test)


# In[39]:


mae = mean_absolute_error(y_test,ypred)

mae


# In[40]:


rmse = np.sqrt(mean_squared_error(y_test,ypred))
rmse


# In[ ]:





# In[41]:


import xgboost as xgb                      #it is used to check the perfprmance of the model
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

model = XGBRegressor(n_estimators=100, learning_rate=0.1)

model.fit(X_train, y_train)

ypred = model.predict(X_test)

mse = mean_squared_error(y_test, ypred)
mse


# In[42]:


from sklearn.model_selection import GridSearchCV     
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# In[43]:


grid_param={
    'n_estimators':[20,30,40],
    'max_depth':[5,10,15]
}


# In[44]:


ranF=RandomForestRegressor(criterion='absolute_error')


# In[45]:


ranF_grid_search=GridSearchCV(ranF,grid_param,cv=5,scoring ='neg_mean_absolute_error')


# In[46]:


ranF_grid_search.fit(x_21,y_21)  


# In[47]:


ypred21=ranF_grid_search.best_estimator_.predict(X_test)
best_model21=ranF_grid_search.best_estimator_
best_model21


# In[48]:


ypred22=ranF_grid_search.best_estimator_.predict(X_test)
best_model22=ranF_grid_search.best_estimator_
best_model22


# In[49]:


from sklearn.metrics import mean_absolute_error,mean_squared_error  #it is used to check the perfprmance of the model


# In[50]:


mean_absolute_error(y_test, ypred22)


# In[51]:


np.sqrt(mean_squared_error(y_test, ypred22))


# In[52]:


ypred22


# In[53]:


mae = mean_absolute_error(y_test,ypred22)

mae


# In[54]:


y_test


# In[55]:


rf.score(X_test,y_test)


# In[56]:


rmse = np.sqrt(mean_squared_error(y_test,ypred22))
rmse


# In[57]:


import pickle   # this wouild save the code


# In[58]:


filename='midsem.pkl'
pickle.dump(best_model21, open(filename,'wb'))


# In[59]:


filename='scaler.pkl'
pickle.dump(scaler, open(filename,'wb'))


# In[62]:


get_ipython().system('pip install streamlit  # this code would deploy it on the web')
import streamlit as st


# In[63]:


filename='midsem.pkl'
pickle.dump(best_model21, open(filename,'wb'))


# In[66]:


get_ipython().system('pip show scikit-learn')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




