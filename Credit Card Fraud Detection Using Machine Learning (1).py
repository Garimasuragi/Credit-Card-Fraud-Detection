#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


cd C:\Users\DELL\Downloads\Telegram Desktop


# In[6]:


credit_card_data=pd.read_csv("creditcard.csv")


# In[7]:


credit_card_data


# In[8]:


credit_card_data.head()


# In[9]:


credit_card_data.tail()


# In[10]:


#data information
credit_card_data.info()


# In[11]:


#checking the number of missing value
credit_card_data.isnull().sum()


# In[12]:


#distribution of legit transactions & fraudulent transactions
credit_card_data["Class"].value_counts()


# In[13]:


#separating the data for analysis
legit=credit_card_data[credit_card_data.Class == 0]
fraud=credit_card_data[credit_card_data.Class == 1]


# In[14]:


print(legit.shape)
print(fraud.shape)


# In[15]:


#statistical measures of the data
legit.Amount.describe()


# In[17]:


fraud.Amount.describe()


# In[18]:


#compare the values for both transactions
credit_card_data.groupby("Class").mean()


# In[20]:


#simple trasaction
legit_sample = legit.sample(n=492)


# In[22]:


n_dataset = pd.concat([legit_sample, fraud],axis=0)


# In[25]:


n_dataset.head()


# In[24]:


n_dataset.tail()


# In[26]:


n_dataset["Class"].value_counts()


# In[28]:


n_dataset.groupby("Class").mean()


# In[29]:


#Splitting the data into features&Targets
X= n_dataset.drop(columns='Class',axis=1)
Y=n_dataset["Class"]


# In[30]:


print(X)


# In[31]:


print(Y)


# In[33]:


#split the data into training data & testing data
X_train, X_test, Y_train, Y_test =train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)


# In[34]:


print(X.shape, X_train.shape,X_test.shape)


# In[ ]:


#Model  Training
#Logistic Regression


# In[35]:


model=LogisticRegression()


# In[36]:


#training the Logistic RegressionModel with Training Data
model.fit(X_train, Y_train)


# In[ ]:


#Model Evaluation
#Acccuracy Score


# In[38]:


#Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[39]:


print("Accuracy on Training data: ", training_data_accuracy)


# In[41]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[43]:


print("Accuracy score on Test data: ", test_data_accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




