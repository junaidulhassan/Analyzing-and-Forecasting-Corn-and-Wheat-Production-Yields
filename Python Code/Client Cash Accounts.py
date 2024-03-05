#!/usr/bin/env python
# coding: utf-8

# ### Station 1 : ETL
# #### `Dataset` : Client Cash Accounts

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wn
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
wn.filterwarnings('ignore')


# #### Load Dataset

# In[2]:


dataset = pd.read_csv('datasets/Client_Cash_Accounts.csv')
dataset.head()


# In[106]:


# change the Date format into pandas datatime formate.
dataset['Transaction Date'] = pd.to_datetime(dataset['Transaction Date'])


# #### Information of dataset

# * Client: The ID of the client associated with the transaction.
# * Transaction Date: The date the transaction occurred.
# * Type: The type of transaction, either "D" for deposit or "W" for withdrawal.
# * Customer / Supplier: The name of the customer or supplier involved in the transaction.
# * Reference: A reference code or number associated with the transaction.
# * Description: A brief description of the transaction.
# * Flow: The amount of money involved in the transaction.
# * Balance: The balance of the account after the transaction has been processed.

# In[66]:


dataset.info()


# #### Description of dataset

# In[67]:


dataset.describe()


# #### Handle missing values

# #### There is no missing values in our dataset

# In[68]:


dataset.isnull().sum()


# #### Check the Duplicate values in every columns

# In[69]:


dataset['Type'].value_counts()


# In[70]:


dataset['Client'].value_counts()


# In[71]:


#duplicate value in this columns
dataset['Customer / Supplier'].value_counts()


# In[72]:


dataset['Reference'].value_counts()


# In[73]:


dataset['Description'].value_counts()


# ### Station 2 : Feature Engineering

# #### Check the Correlation between the flow and balance

# In[89]:


corr_dataset = dataset[['Flow','Balance']]
sns.heatmap(corr_dataset.corr(),annot=True)


# * The plot shows a color-coded matrix where each cell represents the correlation between two variables. 
# * The diagonal line represents the correlation between a variable and itself, which is always 1
# * If the correlation coefficient is close to 1, it indicates a strong positive correlation.
# * If it is close to -1, it indicates a strong negative correlation.

# In[101]:


# check the Distribution of cash flows
sns.distplot(dataset["Flow"],kde=True,hist=False,color='r')
plt.title("Distribution of Flows")
plt.xlabel("Flow")
plt.show()


# * The plot shows the frequency of different values of flow, indicating the distribution. 
# * The plot is customized to show only the kernel density estimate (kde)

# In[75]:


# Times of balance of client
plt.figure(figsize=(10,5))
sns.lineplot(x="Transaction Date", y="Balance", data=dataset,color='Green')
plt.title("Time Series of Balance")
plt.xlabel("Transaction Date")
plt.ylabel("Balance")
plt.xticks(rotation=80)
plt.show()


# * This plot shows the results of a time series analysis of the transaction amounts. 
# * It is used to identify any trends or patterns in the data over time, such as seasonality or trends.

# In[76]:


# Client type over the time of balance of client
plt.figure(figsize=(10,5))
sns.scatterplot(x="Transaction Date", y="Balance", data=dataset,hue=dataset['Type'])
plt.title("Type Over Time Series of Balance")
plt.xlabel("Transaction Date")
plt.ylabel("Balance")
plt.xticks(rotation=80)
plt.show()


# * The x-axis represents the transaction date, while the y-axis represents the balance of the client account. 
# * The hue parameter is used to color-code the points based on the transaction type (deposit vs. withdrawal).
# * Each point represents a transaction, and the color of the point indicates the transaction type. 

# In[77]:


# check the Transaction Types 
sns.countplot(x="Type", data=dataset)
plt.title("Bar Chart of Transaction Types")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.show()


# * Number of clients (Diposit vs Withdraw)

# In[78]:


# Check the outliters in our dataset using boxplot
sns.boxplot(x=dataset["Flow"])
plt.title("Boxplot of Flows")
plt.xlabel("Flow")
plt.show()


# * The plot shows a box that represents the interquartile range (IQR) of the data, with the median value marked by a horizontal line inside the box. 
# * The whiskers of the plot extend to the minimum and maximum values within the IQR multiplied by an allowable range (usually 1.5 times the IQR). 
# * Any data points outside of the whiskers are considered to be outliers and are represented by individual points.

# In[79]:


# Check the outliters on basis of type of client
sns.boxplot(x="Type", y="Flow", data=dataset)
plt.title("Boxplot of Flows by Type")
plt.xlabel("Transaction Type")
plt.ylabel("Flow")
plt.show()


# * The plot shows two boxes side by side, one for each transaction type, with the median value marked by a horizontal line inside the box. 
# * The whiskers of the plot extend to the minimum and maximum values within the IQR multiplied by an allowable range (usually 1.5 times the IQR). 
# * Any data points outside of the whiskers are considered to be outliers and are represented by individual points
# * The plot helps to compare the distribution of cash flows for deposit and withdrawal transactions and identify any extreme values that may be considered outliers for each type of transaction.

# In[80]:


# Transaction amount distribution
plt.figure(figsize=(10,5))
sns.countplot(x='Flow', data=dataset)
plt.xlabel('Transaction Amount')
plt.ylabel('Count')
plt.title('Transaction Amount Distribution')
plt.xticks(rotation=80)
plt.show()


# * Each bar represents a specific transaction amount, and the height of the bar indicates the number of transactions with that amount. 

# In[93]:


plt.figure(figsize=(8,6))
sns.violinplot(x='Type', y='Flow', data=dataset)
plt.xlabel('Transaction Type')
plt.ylabel('Transaction Amount')
plt.title('Transaction Amount by Type')
plt.show()


# * The x-axis represents the transaction type (deposit vs. withdrawal), while the y-axis represents the transaction amount (flow). 
# * Each violin plot shows the distribution of transaction amounts for a specific transaction type, with the width of the violin indicating the density of the data points.

# In[100]:


from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(dataset['Flow'], model='additive', period=1)
fig = decomposition.plot()
plt.show()


# * The decomposition plot helps to visualize the different components of the time series data and understand how they contribute to the overall pattern of the data. 
# * It can be used to identify any trends or seasonal patterns in the data that is useful for for analysis.

# ### Data Preprocessing
# * One Hot encoding
# * Label Encoding

# In[119]:


# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(dataset['Type'])
dataset['Type'] = le.fit_transform(dataset['Type'])


# In[120]:


dataset.head()


# In[121]:


# OneHot Encoding
categorical_columns = ['Customer / Supplier', 'Reference', 'Description']
dataset = pd.get_dummies(data=dataset, columns=categorical_columns, dtype=int)


# In[122]:


dataset = dataset.drop(['Transaction Date','Client'],axis=1)
dataset.head()


# ### Lilnear Regression Analysis

# In[125]:


# seperate the target and independent variable
x = dataset.drop('Balance',axis=1)
y = dataset['Balance']


# In[126]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[139]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x,y)


# In[140]:


acc = model.score(x_test,y_test)
print('The accuracy of the model is ',acc)


# In[ ]:




