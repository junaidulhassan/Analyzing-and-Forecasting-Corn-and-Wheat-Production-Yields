#!/usr/bin/env python
# coding: utf-8

# ## Station 1 : ETL
# #### Dataset : Corn Price History

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings as wn
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
wn.filterwarnings('ignore')


# #### Load dataset

# In[36]:


dataset = pd.read_csv('datasets/Corn_price_history.csv')
dataset.head()


# #### Dataset Description

# In[4]:


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

# In[5]:


dataset.info()


# #### Handle Missing values

# In[6]:


dataset.isnull().sum()*100/len(dataset)


# In[7]:


dataset.info()


# In[8]:


# we need to drop null values in dataset
dataset = dataset.dropna()
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%m/%d/%y')
dataset.head()


# ## Station 2 : Feature Engineering

# In[9]:


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

# In[10]:


plt.figure(figsize=(10, 6))
sns.histplot(x='Settlement Price', data=dataset, bins=20)
plt.title('Distribution of Wheat Settlement Prices')
plt.xlabel('Settlement Price')
plt.ylabel('Count')
plt.show()


# * The graph represents the distribution of the settlement prices for corn futures. 
# * The x-axis shows the range of settlement prices, divided into 20 bins, and the y-axis shows the count of how many times a given settlement price appears in the dataset.

# In[11]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Last', y='Settlement Price', data=dataset)
plt.title('Last Price vs. Settlement Price')
plt.xlabel('Last Price')
plt.ylabel('Settlement Price')
plt.show()


# * The scatter plot provides a visual representation of the relationship between the 'Last' and 'Settlement 
# *  Price' features, showing whether there is a positive, negative, or no correlation between the two features.
# * Each dot represents a single data point, and the position of the dot on the plot indicates the value of the 'Last' feature on the x-axis and the value of the 'Settlement Price' feature on the y-axis. 
# * The scatter plot can be used to identify any patterns or trends in the relationship between the two features, as well as any potential outliers or unusual data points.

# In[12]:


from statsmodels.tsa.seasonal import seasonal_decompose

dataset['Date'] = pd.to_datetime(dataset['Date'], format='%m/%d/%y')
dataset.set_index('Date', inplace=True)

result = seasonal_decompose(dataset['Settlement Price'], model='additive', period=30)
result.plot()
plt.show()


# * The resulting plot shows four subplots: the original time-series data, the trend component, the seasonal component, and the residual component. 
# * The decomposition allows us to visually identify the underlying patterns and fluctuations in the data, and can be used to detect any seasonal or trend-based patterns that may be present in the data.

# In[13]:


# violin plot
dataset['Year'] = dataset.index.year
plt.figure(figsize=(10, 6))
sns.violinplot(x='Year', y='Settlement Price', data=dataset)
plt.title('Distribution of Wheat Settlement Prices by Year')
plt.xlabel('Year')
plt.ylabel('Settlement Price')
plt.show()


# * The violin plot provides a visual representation of how the distribution of the settlement prices varies across different years, showing whether there are any significant differences in the shape of the distribution or the range of prices between different years. 
# * The width of each curve indicates the relative frequency of the settlement prices at each point along the y-axis, with wider curves indicating a higher frequency of prices in that range. 
# * The violin plot can be used to compare the distribution of the settlement prices between different years and identify any trends or patterns in the data.

# In[14]:


dataset['Month'] = dataset.index.month
plt.figure(figsize=(10, 6))
sns.boxplot(x='Month', y='Settlement Price', data=dataset)
plt.title('Box Plot of Wheat Settlement Prices by Month')
plt.xlabel('Month')
plt.ylabel('Settlement Price')
plt.show()


# * The box plot provides a visual representation of how the distribution of the settlement prices varies across different months, showing the median, quartiles, and outliers for each month. 
# * The boxes indicate the range of the middle 50% of the settlement prices for each month, with the line inside the box indicating the median value. 
# * The whiskers extend to the minimum and maximum values within 1.5 times the interquartile range of the box, and any points outside the whiskers are considered outliers. 
# * The box plot can be used to compare the distribution of the settlement prices between different months and identify any trends or patterns in the data.

# In[15]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='CVol', y='Settlement Price', data=dataset,c='#FF8D33')
plt.title('Trading Volume vs. Settlement Price')
plt.xlabel('Trading Volume')
plt.ylabel('Settlement Price')
plt.show()


# * The scatter plot provides a visual representation of the relationship between the 'CVol' and 'Settlement Price' features, showing whether there is a positive, negative, or no correlation between the two features. 
# * Each dot represents a single data point, and the position of the dot on the plot indicates the value of the 'CVol' feature on the x-axis and the value of the 'Settlement Price' feature on the y-axis. 
# * The scatter plot can be used to identify any patterns or trends in the relationship between the two features, as well as any potential outliers or unusual data points.

# In[16]:


dataset['Price Change'] = dataset['Settlement Price'] - dataset['Settlement Price'].shift(1)

plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Price Change', data=dataset)
plt.title('Mean Price Change by Year')
plt.xlabel('Year')
plt.ylabel('Mean Price Change')
plt.show()


# * The bar plot provides a visual representation of the mean change in the settlement prices from one day to the next across different years, showing whether there are any significant differences in the price changes between different years. 
# * Each bar represents the mean price change for a single year, and the height of the bar indicates the magnitude of the mean price change. 
# * The bar plot can be used to compare the mean price changes between different years and identify any trends or patterns in the data.

# In[17]:


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

# In[18]:


dataset['Open Interest'] = dataset['Open Interest'].str.replace(',','')
dataset['Open Interest']=dataset['Open Interest'].astype(float)


# ### Correlation between Attributes

# In[19]:


plt.figure(figsize=(12,8))
sns.heatmap(dataset.corr(),annot=True)


# * Correlation measures the degree to which two attributes are related to each other.
# * Positive correlation means two attributes tend to increase or decrease together, negative correlation means they tend to move in opposite directions, and neutral correlation means there is no relationship between the attributes.
# * Correlation can be measured using statistical techniques such as the Pearson correlation coefficient, which ranges from -1 to 1.
# * Correlation can help understand the relationship between attributes and detect patterns and trends in the data.
# * Correlation does not imply causation, which means one attribute does not necessarily cause the other even if they are correlated.
# * Correlation can be influenced by outliers, the scale of the data, and the sample size, so it is important to use multiple methods to analyze and interpret data.

# In[20]:


df_var = dataset.drop('Settlement Price',axis=1)
df_target = dataset['Settlement Price']


# ### Drop the columns which are highly correlated to each other

# In[21]:


corr_matrix = df_var.corr().abs()

# Create a set to keep track of dropped columns
dropped_cols = set()

# Iterate through the correlation matrix and drop correlated columns
for col in corr_matrix.columns:
    if col not in dropped_cols:
        correlated_cols = corr_matrix.index[corr_matrix[col] > 0.98].tolist()
        correlated_cols.remove(col)  # Remove the current column from the list of correlated columns
        df_var.drop(correlated_cols, axis=1, inplace=True)
        dropped_cols.update(correlated_cols)


# In[22]:


sns.heatmap(df_var.corr(),annot=True)


# ### Predictive Model

# In[23]:


dataset.info()


# #### Set the target value as a settlement Price 

# In[24]:


dataset = dataset.dropna()


# In[25]:


x = dataset.drop('Settlement Price',axis=1)
y = dataset['Settlement Price']


# In[26]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# ### Linear Regression model
# Apply the machine learning model to predict the settlement prices

# In[27]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[28]:


sns.lineplot(model.coef_)
plt.title('Loss Function')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()


# * This graph show the how rate of loss or error reduce during training of machine learning model

# In[32]:


acc = model.score(x_test,y_test)
print('The accuracy of the model is ',acc*100,'%')


# ### Mean Square Error Analysis

# In[33]:


y_pred = model.predict(x_test)


# In[34]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)


# In[35]:


print("The Mean square the error of the model is ", mse)

