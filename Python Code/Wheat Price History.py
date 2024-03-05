#!/usr/bin/env python
# coding: utf-8

# ## Station 1 : ETL
# #### Dataset : Wheat Price History

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings as wn
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
wn.filterwarnings('ignore')


# #### Load dataset

# In[12]:


dataset = pd.read_csv('datasets/Wheat_price_history.csv')
dataset.head()


# #### Dataset Description

# In[13]:


dataset.describe()


# * `Date`: The date of the trading day for the corn futures contract.
# * `Last`: The last traded price for the corn futures contract on that trading day.
# * `Settlement Price`: The official settlement price for the corn futures contract on that trading day.
# * `Change`: The change in the Last price from the previous trading day.
# * `% Change`: The percentage change in the Last price from the previous trading day.
# * `Bid`: The highest price a buyer is willing to pay for the corn futures contract on that trading day.
# * `Ask`: The lowest price a seller is willing to accept for the corn futures contract on that trading day.
# * `Open` Interest: The total number of outstanding contracts for the corn futures contract as of the end of that trading day.
# * `CVol`: The volume of contracts traded during that trading day.
# * `Open`: The opening price for the corn futures contract on that trading day.
# * `High`: The highest price reached by the corn futures contract during that trading day.
# * `Low`: The lowest price reached by the corn futures contract during that trading day.

# #### Dataset Info

# In[14]:


dataset.info()


# #### Handle Missing values

# In[15]:


dataset.isnull().sum()*100/len(dataset)


# In[16]:


dataset.info()


# In[17]:


# we need to drop null values in dataset
dataset = dataset.dropna()
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%m/%d/%y')
dataset.head()


# ## Station 2 : Feature Engineering

# In[18]:


sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))
# Format x-axis date ticks
sns.lineplot(x='Date', y='Settlement Price', data=dataset,c='red')
#sns.barplot(x='Date', y='Settlement Price', data=dataset)
plt.title('Wheat Settlement Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Settlement Price')
plt.xticks(rotation=80)
plt.show()


# * The graph represents the historical trend of the settlement prices for corn futures over time. 
# * The x-axis shows the dates on which the corn futures were traded, and the y-axis shows the corresponding settlement prices for those trading days. 

# In[19]:


plt.figure(figsize=(10, 6))
sns.histplot(x='Settlement Price', data=dataset, bins=20,color='r')
plt.title('Distribution of Wheat Settlement Prices')
plt.xlabel('Settlement Price')
plt.ylabel('Count')
plt.show()


# * The graph represents the distribution of the settlement prices for wheat futures. 
# * The x-axis shows the range of settlement prices, divided into 20 bins, and the y-axis shows the count of how many times a given settlement price appears in the dataset.

# In[20]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Last', y='Settlement Price', data=dataset)
plt.title('Last Price vs. Settlement Price')
plt.xlabel('Last Price')
plt.ylabel('Settlement Price')
plt.show()


# * The scatter plot provides a visual representation of the relationship between the 'Last' and 'Settlement 
# * Price' features, showing whether there is a positive, negative, or no correlation between the two features.
# * Each dot represents a single data point, and the position of the dot on the plot indicates the value of the 'Last' feature on the x-axis and the value of the 'Settlement Price' feature on the y-axis. 
# * The scatter plot can be used to identify any patterns or trends in the relationship between the two features, as well as any potential outliers or unusual data points.

# In[21]:


from statsmodels.tsa.seasonal import seasonal_decompose

dataset['Date'] = pd.to_datetime(dataset['Date'], format='%m/%d/%y')
dataset.set_index('Date', inplace=True)

result = seasonal_decompose(dataset['Settlement Price'], model='additive', period=30)
result.plot()
plt.show()


# * The resulting plot shows four subplots: the original time-series data, the trend component, the seasonal component, and the residual component. 
# * The decomposition allows us to visually identify the underlying patterns and fluctuations in the data, and can be used to detect any seasonal or trend-based patterns that may be present in the data.

# In[22]:


dataset['Year'] = dataset.index.year
plt.figure(figsize=(10, 6))
sns.violinplot(x='Year', y='Settlement Price', data=dataset)
plt.title('Distribution of Wheat Settlement Prices by Year')
plt.xlabel('Year')
plt.ylabel('Settlement Price')
plt.show()


# * The violin plot shows the distribution of wheat settlement prices for each year in the dataset. 
# * The x-axis shows the years, while the y-axis shows the settlement prices. 
# * The width of each violin represents the density of the data for that year, with wider violins representing more data. 
# * The violin plot can be useful for comparing the distribution of settlement prices between different years, and identifying any patterns or trends in the data.

# In[24]:


dataset['Month'] = dataset.index.month
plt.figure(figsize=(10, 6))
sns.boxplot(x='Month', y='Settlement Price', data=dataset)
plt.title('Box Plot of Wheat Settlement Prices by Month')
plt.xlabel('Month')
plt.ylabel('Settlement Price')
plt.show()


# * The box plot shows the distribution of wheat settlement prices for each month in the dataset. 
# * The x-axis shows the months, while the y-axis shows the settlement prices. 
# * Each box represents the interquartile range (IQR) of the data for that month, with the whiskers extending to the minimum and maximum values within 1.5 times the IQR from the first and third quartiles, respectively. 
# * Any data points beyond the whiskers are considered outliers and are plotted as individual points. 
# * The box plot can be useful for comparing the distribution of settlement prices between different months, and identifying any patterns or trends in the data.

# In[25]:


#scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='CVol', y='Settlement Price', data=dataset,c='#FF8D33')
plt.title('Trading Volume vs. Settlement Price')
plt.xlabel('Trading Volume')
plt.ylabel('Settlement Price')
plt.show()


# * The scatter plot shows the relationship between the trading volume and the settlement price for the dataset. 
# * Each data point represents a single observation, with the x-coordinate representing the trading volume and the y-coordinate representing the settlement price. 
# * The color of the data points is set to '#FF8D33'. 
# * The scatter plot can be useful for identifying any correlation or pattern between the two variables.

# In[26]:


dataset['Price Change'] = dataset['Settlement Price'] - dataset['Settlement Price'].shift(1)
plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Price Change', data=dataset)
plt.title('Mean Price Change by Year')
plt.xlabel('Year')
plt.ylabel('Mean Price Change')
plt.show()


# * The bar plot shows the mean price change for each year in the dataset. 
# * The x-axis shows the years, while the y-axis shows the mean price change. 
# * The height of each bar represents the mean price change for that year. 
# * The bar plot can be useful for comparing the mean price change between different years, and identifying any patterns or trends in the data. 

# In[27]:


dataset['Bid-Ask Spread'] = dataset['Ask'] - dataset['Bid']
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Bid-Ask Spread', y='Settlement Price', data=dataset)
plt.title('Bid-Ask Spread vs. Settlement Price')
plt.xlabel('Bid-Ask Spread')
plt.ylabel('Settlement Price')
plt.show()


# * The scatter plot provides a visual representation of the relationship between the 'Bid-Ask Spread' and 'Settlement Price' features, showing whether there is a positive, negative, or no correlation between the two features. 
# * Each dot represents a single data point, and the position of the dot on the plot indicates the value of the 'Bid-Ask Spread' feature on the x-axis and the value of the 'Settlement Price' feature on the y-axis. 
# * The scatter plot can be used to identify any patterns or trends in the relationship between the two features, as well as any potential outliers or unusual data points.

# In[28]:


dataset['Open Interest'] = dataset['Open Interest'].str.replace(',','')
dataset['Open Interest']=dataset['Open Interest'].astype(float)


# ### Correlation between the Features

# In[29]:


plt.figure(figsize=(12,8))
sns.heatmap(dataset.corr(),annot=True)


# * Correlation measures the degree to which two attributes are related to each other.
# * Positive correlation means two attributes tend to increase or decrease together, negative correlation means they tend to move in opposite directions, and neutral correlation means there is no relationship between the attributes.
# * Correlation can be measured using statistical techniques such as the Pearson correlation coefficient, which ranges from -1 to 1.
# * Correlation can help understand the relationship between attributes and detect patterns and trends in the data.
# * Correlation does not imply causation, which means one attribute does not necessarily cause the other even if they are correlated.
# * Correlation can be influenced by outliers, the scale of the data, and the sample size, so it is important to use multiple methods to analyze and interpret data.

# ### Delete the high correlated features

# In[30]:


df_var = dataset.drop('Settlement Price',axis=1)
df_target = dataset['Settlement Price']

corr_matrix = df_var.corr().abs()

# Create a set to keep track of dropped columns
dropped_cols = set()

# Iterate through the correlation matrix and drop correlated columns
for col in corr_matrix.columns:
    if col not in dropped_cols:
        correlated_cols = corr_matrix.index[corr_matrix[col] > 0.99].tolist()
        correlated_cols.remove(col)  # Remove the current column from the list of correlated columns
        df_var.drop(correlated_cols, axis=1, inplace=True)
        dropped_cols.update(correlated_cols)


# ### After delete features

# In[31]:


plt.figure(figsize=(10,7))
sns.heatmap(df_var.corr(),annot=True)


# ### Predictive Model

# In[32]:


dataset.info()


# #### Set the target value as a settlement Price 

# In[33]:


dataset = dataset.dropna()


# In[34]:


x = dataset.drop('Settlement Price',axis=1)
y = dataset['Settlement Price']


# In[35]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# ### Linear Regression model

# In[36]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[37]:


sns.lineplot(model.coef_)
plt.title('Loss Function')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()


# * The line plot shows the loss function of a machine learning model over multiple epochs of training. 
# * The x-axis shows the epoch number, while the y-axis shows the loss value. 
# * The line plot can be useful for monitoring the performance of the model during training and identifying any patterns or trends in the loss function over time.

# In[38]:


acc = model.score(x_test,y_test)
print('The accuracy of the model is ',acc,'%')


# ### Mean Square Error Analysis

# In[39]:


y_pred = model.predict(x_test)


# In[40]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)


# In[41]:


print("The Mean square the error of the model is ", mse)

