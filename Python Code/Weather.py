#!/usr/bin/env python
# coding: utf-8

# ## Station 1 ETL
# #### Dataset : Wheather Condition
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wn
wn.filterwarnings('ignore')


# #### Load Dataset

# In[2]:


dataset = pd.read_csv('datasets/Weather (2).csv')
dataset.head()


# #### Information about dataset

# In[3]:


dataset.info()


# #### Description of dataset 

# In[4]:


dataset.describe()


# * Date: Date of the recorded weather data
# * Minimum temperature +AKk-: Minimum temperature in Celsius (°C) for the day
# * Maximum temperature (C): Maximum temperature in Celsius (°C) for the day
# * Rainfall (mm): Amount of rainfall in millimeters (mm) for the day
# * Direction of maximum wind gust: Direction of the maximum wind gust recorded for the day
# * Speed of maximum wind gust (km/h): Speed of the maximum wind gust recorded in kilometers per hour (km/h) for the day
# * Time of maximum wind gust: Time of the day when the maximum wind gust was recorded
# * 9am Temperature (C): Temperature in Celsius (°C) recorded at 9am
# * 9am relative humidity (+ACU-): Relative humidity recorded at 9am, expressed as a percentage
# * 9am wind direction: Wind direction recorded at 9am
# * 9am wind speed (km/h): Wind speed recorded at 9am in kilometers per hour (km/h)
# * 3pm Temperature (C): Temperature in Celsius (°C) recorded at 3pm
# * 3pm relative humidity (+ACU-): Relative humidity recorded at 3pm, expressed as a percentage
# * 3pm wind direction: Wind direction recorded at 3pm
# * 3pm wind speed (km/h): Wind speed recorded at 3pm in kilometers per hour (km/h)

# ### Handle the missing values

# In[5]:


# Calculate the percentage of missing values for each column

missing_pct = dataset.isnull().sum() * 100 / len(dataset)
plt.figure(
    figsize=(14,7)
)
# Create a bar plot of the missing value percentage for each column
ax = sns.barplot(x=missing_pct.index, y=missing_pct)
plt.title("Percentage of Missing Values by Column")
plt.xlabel("Column")
plt.ylabel("Percentage Missing (%)")
plt.xticks(rotation=80)

# Add percentage labels to the bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()


# * This barplot show the percentage of the missing values in every columns

# In[6]:


# drop the missing values rows 
dataset = dataset.dropna()
missing_pct = dataset.isnull().sum() * 100 / len(dataset)
missing_pct


# ## Station 2 Feature Engineering
# #### Dataset : Wheather Condition 

# In[39]:


#Distribution graph
sns.distplot(dataset['Minimum temperature +AKk-'],kde=True,hist=False)
sns.distplot(dataset['Maximum temperature (C)'],kde=True,hist=False);


# * The distribution plot shows the distribution of values in each column as a histogram, with the x-axis representing the range of values and the y-axis representing the frequency or density of those values. 
# * The plot has a curve that shows the estimated probability density function of the data.

# ### Correlation between features

# In[8]:


non_object_columns = dataset.select_dtypes(exclude=['object'])


# In[9]:


sns.heatmap(non_object_columns.corr(),
            annot=True,cmap='coolwarm')


# In[37]:


sns.scatterplot(x='Maximum temperature (C)', y='Rainfall (mm)', data=dataset)
plt.title('Maximum Temperature vs Rainfall')
plt.show()

# Histogram of 9am temperature
sns.histplot(data=dataset, x='9am Temperature (C)', bins=10)
plt.title('9am Temperature Histogram')
plt.show()

# Box plot of maximum wind gust speed by wind direction
sns.boxplot(x='Direction of maximum wind gust ', y='Speed of maximum wind gust (km/h)', data=dataset)
plt.title('Maximum Wind Gust Speed by Wind Direction')
plt.show()

# Line plot of minimum and maximum temperatures over time
plt.plot(dataset['Date'], dataset['Minimum temperature +AKk-'], label='Minimum Temperature')
plt.plot(dataset['Date'], dataset['Maximum temperature (C)'], label='Maximum Temperature')
plt.title('Minimum and Maximum Temperatures Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=80)
plt.legend()
plt.show()


# ### Add some new useful features

# In[11]:


dataset['Date'] = pd.to_datetime(dataset['Date'], format='%d/%m/%Y')


# In[12]:


dataset['Month'] = dataset['Date'].dt.month

# Create a new column for day of the week
dataset['Day of Week'] = dataset['Date'].dt.dayofweek


# In[13]:


dataset['Wind Speed Category'] = pd.cut(dataset['Speed of maximum wind gust (km/h)'], 
                                     bins=[0, 10, 20, 30, 40, 50, 60, 70, 80],
                                     labels=['0-10', '10-20', '20-30', '30-40', 
                                             '40-50', '50-60', '60-70', '70-80'])

# Create a new column for temperature category
dataset['Temperature Category'] = pd.cut(dataset['Maximum temperature (C)'],
                                      bins=[0, 10, 20, 30, 40], 
                                      labels=['0-10', '10-20', '20-30', '30-40'])


# In[14]:


# Calculate the total rainfall by month
total_rainfall = dataset.groupby('Month')['Rainfall (mm)'].sum()

# Calculate the average wind speed by wind direction
avg_wind_speed = dataset.groupby('Direction of maximum wind gust ')['Speed of maximum wind gust (km/h)'].mean()


# In[15]:


sns.barplot(x=total_rainfall.index, y=total_rainfall.values)
plt.title('Total Rainfall by Month')
plt.xlabel('Month')
plt.ylabel('Total Rainfall (mm)')
plt.show()


# * The plot shows the total rainfall in millimeters for each month of the year, as represented by the height of the bars. 
# * The x-axis shows the months of the year, while the y-axis shows the total rainfall in millimeters. 
# * The bar plot can be useful for comparing the total rainfall between different months, and identifying any patterns or trends in the data.

# In[16]:


# Create a bar chart of average wind speed by wind direction
sns.barplot(x=avg_wind_speed.index, y=avg_wind_speed.values)
plt.title('Average Wind Speed by Wind Direction')
plt.xlabel('Wind Direction')
plt.ylabel('Average Wind Speed (km/h)')
plt.show()


# * The plot shows the average wind speed in kilometers per hour for each wind direction, as represented by the height of the bars. 
# * The x-axis shows the wind directions, while the y-axis shows the average wind speed in kilometers per hour. 
# * The bar plot can be useful for comparing the average wind speed between different wind directions, and identifying any patterns or trends in the data.

# In[17]:


# Create a scatter plot of maximum temperature vs wind speed
sns.scatterplot(x='Maximum temperature (C)', y='Speed of maximum wind gust (km/h)', data=dataset,c='brown')
plt.title('Maximum Temperature vs Wind Speed')
plt.show()


# * The scatter plot shows the relationship between the maximum temperature and the speed of the maximum wind gust recorded for each day in the dataset. 
# * Each data point represents a single day, with the x-coordinate representing the maximum temperature and the y-coordinate representing the speed of the maximum wind gust. 
# * The color of the data points is set to brown. The scatter plot can be useful for identifying any correlation or pattern between the two variables.

# In[18]:


# Create a bar chart of temperature category by day of the week
sns.countplot(x='Day of Week', hue='Temperature Category', data=dataset)
plt.title('Temperature Category by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Count')
plt.show()


# * The count plot shows the number of days in each temperature category for each day of the week. 
# * The x-axis shows the days of the week, while the y-axis shows the count of days. 
# * The bars are grouped by color, with each color representing a different temperature category. 
# * The count plot can be useful for comparing the distribution of temperature categories across different days of the week, and identifying any patterns or trends in the data.

# In[19]:


# Create a bar chart of wind speed category by month
sns.countplot(x='Month', hue='Wind Speed Category', data=dataset)
plt.title('Wind Speed Category by Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.show()


# * The count plot shows the number of days in each wind speed category for each month of the year. 
# * The x-axis shows the months of the year, while the y-axis shows the count of days. 
# * The bars are grouped by color, with each color representing a different wind speed category. 
# * The count plot can be useful for comparing the distribution of wind speed categories across different months of the year, and identifying any patterns or trends in the data.

# ### Rainfall Prediction

# In[20]:


# drop some unneccessary columns
data = dataset.drop(['Date','Wind Speed Category','Temperature Category','Time of maximum wind gust'],axis=1)


# In[21]:


# use label encoder to label the directions
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(data['3pm wind direction'])
data['3pm wind direction label'] = le.fit_transform(data['3pm wind direction'])


# In[22]:


le.fit(data['9am wind direction'])
data['9am wind direction label'] = le.fit_transform(data['9am wind direction'])


# In[23]:


le.fit(data['Direction of maximum wind gust '])
data['Direction of maximum wind gust label'] = le.fit_transform(data['Direction of maximum wind gust '])


# In[24]:


data.head()


# In[25]:


Direction_label = data[['Direction of maximum wind gust ','Direction of maximum wind gust label',
      '3pm wind direction','3pm wind direction label',
     '9am wind direction','9am wind direction label']]
Direction_label.head()


# In[26]:


data = data.drop(['Direction of maximum wind gust ','3pm wind direction','9am wind direction'],axis=1)


# In[27]:


data['9am wind speed (km/h)'] = data['9am wind speed (km/h)'].replace('Calm',None)


# In[28]:


data['9am wind speed (km/h)'] = data['9am wind speed (km/h)'].fillna(data['9am wind speed (km/h)'].median)


# In[29]:


data['3pm wind speed (km/h)'] = data['3pm wind speed (km/h)'].replace('Calm',None)
data['3pm wind speed (km/h)'] = data['3pm wind speed (km/h)'].fillna(data['3pm wind speed (km/h)'].median)


# In[30]:


data = data.drop(['3pm wind speed (km/h)','9am wind speed (km/h)'],axis=1)


# In[31]:


data


# In[32]:


data.info()


# In[33]:


# split the dataset into x and y 
x = data.drop('Rainfall (mm)',axis=1)
y = data['Rainfall (mm)']


# In[34]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[35]:


from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor(n_estimators=1000)
model2.fit(x,y)


# In[38]:


acc = model2.score(x_test,y_test)
print('The accuracy of the model is ',acc*100,'%')


# In[ ]:




