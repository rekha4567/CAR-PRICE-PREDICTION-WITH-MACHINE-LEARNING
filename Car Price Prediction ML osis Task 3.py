#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("CarPrice_Assignment.csv")
data.head()


# In[3]:


data.isnull().sum()


# In[4]:


data.info()


# In[5]:


print(data.describe())


# In[6]:


data.CarName.unique()


# In[7]:


# The price column in this dataset is supposed to be the column whose values we need to predict. So let’s see the distribution of the values of the price column:
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
sns.distplot(data.price)
plt.show()


# In[8]:


#  Now let’s have a look at the correlation among all the features of this dataset:
print(data.corr())


# In[9]:


# Now let’s have a look at the correlation among all the features of this dataset:
plt.figure(figsize=(20, 15))
correlations = data.corr()
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()


# In[10]:


# used the decision tree regression algorithm to train a car price prediction model. So let’s split the data into training and test sets and use the decision tree regression algorithm to train the model:
predict = "price"
data = data[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

from sklearn.metrics import mean_absolute_error
model.score(xtest, predictions)


# - The model gives 100% accuracy on the test set, which is excellent.

# In[ ]:




