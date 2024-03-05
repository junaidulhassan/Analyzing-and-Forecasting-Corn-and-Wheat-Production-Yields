import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wn
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

### STATION 1 : ETL 

# Suppress warnings
wn.filterwarnings('ignore')

# Load Wheat Price History dataset
wheat_dataset = pd.read_csv('datasets/Wheat_Price_History.csv')

# Load Weather dataset
weather_dataset = pd.read_csv('datasets/Weather.csv')

# Load Corn Price History dataset
corn_dataset = pd.read_csv('datasets/Corn_Price_History.csv')

# Load Client Cash Accounts dataset
client_cash_dataset = pd.read_csv('datasets/Client_Cash_Accounts.csv')

# Convert date column to pandas datetime format
client_cash_dataset['Transaction Date'] = pd.to_datetime(client_cash_dataset['Transaction Date'])
wheat_dataset['Date'] = pd.to_datetime(dataset['Date'], format='%m/%d/%y')
corn_dataset['Date'] = pd.to_datetime(dataset['Date'], format='%m/%d/%y')
weather_dataset['Date'] = pd.to_datetime(dataset['Date'], format='%m/%d/%y')

# description of dataset
wheat_dataset.describe()
corn_dataset.describe()
weather_dataset.describe()

# Plot missing value percentage for each column in Wheat Price History dataset
missing_pct = wheat_dataset.isnull().sum() * 100 / len(wheat_dataset)
plt.figure(figsize=(14,7))
ax = sns.barplot(x=missing_pct.index, y=missing_pct)
plt.title("Percentage of Missing Values by Column - Wheat Price History Dataset")
plt.xlabel("Column")
plt.ylabel("Percentage Missing (%)")
plt.xticks(rotation=80)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.show()

# Plot missing value percentage for each column in Weather dataset
missing_pct = weather_dataset.isnull().sum() * 100 / len(weather_dataset)
plt.figure(figsize=(14,7))
ax = sns.barplot(x=missing_pct.index, y=missing_pct)
plt.title("Percentage of Missing Values by Column - Weather Dataset")
plt.xlabel("Column")
plt.ylabel("Percentage Missing (%)")
plt.xticks(rotation=80)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.show()

# Plot missing value percentage for each column in Corn Price History dataset
missing_pct = corn_dataset.isnull().sum() * 100 / len(corn_dataset)
plt.figure(figsize=(14,7))
ax = sns.barplot(x=missing_pct.index, y=missing_pct)
plt.title("Percentage of Missing Values by Column - Corn Price History Dataset")
plt.xlabel("Column")
plt.ylabel("Percentage Missing (%)")
plt.xticks(rotation=80)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.show()

# Plot missing value percentage for each column in Client Cash Accounts dataset
missing_pct = client_cash_dataset.isnull().sum() * 100 / len(client_cash_dataset)
plt.figure(figsize=(14,7))
ax = sns.barplot(x=missing_pct.index, y=missing_pct)
plt.title("Percentage of Missing Values by Column - Client Cash Accounts Dataset")
plt.xlabel("Column")
plt.ylabel("Percentage Missing (%)")
plt.xticks(rotation=80)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.show()


### STATION : 2 FEATURE ENGINEERING



# Visualize Wheat Price History dataset
plt.figure(figsize=(10,5))
sns.lineplot(x="Date", y="Price", data=wheat_dataset, color='blue')
plt.title("Time Series of Wheat Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=80)
plt.show()

# Visualize Weather dataset
plt.figure(figsize=(10,5))
sns.lineplot(x="Date", y="Temperature", data=weather_dataset, color='red')
plt.title("Time Series of Temperature")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.xticks(rotation=80)
plt.show()

# Visualize Corn Price History dataset
plt.figure(figsize=(10,5))
sns.lineplot(x="Date", y="Last", data=corn_dataset, color='green')
plt.title("Time Series of Corn Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=80)
plt.show()

# Visualize Client Cash Accounts dataset
plt.figure(figsize=(10,5))
sns.lineplot(x="Transaction Date", y="Balance", data=client_cash_dataset, color='purple')
plt.title("Time Series of Balance")
plt.xlabel("Transaction Date")
plt.ylabel("Balance")
plt.xticks(rotation=80)
plt.show()

# Plot correlation heatmap for Flow and Balance columns in Client Cash Accounts dataset
corr_dataset = client_cash_dataset[['Flow', 'Balance']]
sns.heatmap(corr_dataset.corr(), annot=True)
plt.title('Correlation Heatmap for Client Cash Accounts Dataset')
plt.show()

# Plot correlation heatmap for High and Low columns in Wheat Price History dataset
corr_dataset = wheat_dataset
sns.heatmap(corr_dataset.corr(), annot=True)
plt.title('Correlation Heatmap for Wheat Price History Dataset')
plt.show()

# Plot correlation heatmap for Precipitation and Temperature columns in Weather dataset
corr_dataset = weather_dataset
sns.heatmap(corr_dataset.corr(), annot=True)
plt.title('Correlation Heatmap for Weather Dataset')
plt.show()

# Plot correlation heatmap for Last and Open columns in Corn Price History dataset
corr_dataset = corn_dataset
sns.heatmap(corr_dataset.corr(), annot=True)
plt.title('Correlation Heatmap for Corn Price History Dataset')
plt.show()

# Plot distribution of cash flows in client_cash_dataset
sns.distplot(client_cash_dataset["Flow"], kde=True, hist=False, color='r')
plt.title("Distribution of Flows")
plt.xlabel("Flow")
plt.show()

# Plot boxplot of flows in client_cash_dataset
sns.boxplot(x=client_cash_dataset["Flow"])
plt.title("Boxplot of Flows")
plt.xlabel("Flow")
plt.show()

# Plot boxplot of flows by transaction type in client_cash_dataset
sns.boxplot(x="Type", y="Flow", data=client_cash_dataset)
plt.title("Boxplot of Flows by Type")
plt.xlabel("Transaction Type")
plt.ylabel("Flow")
plt.show()

# Plot count plot of transaction types in client_cash_dataset
sns.countplot(x="Type", data=client_cash_dataset)
plt.title("Bar Chart of Transaction Types")
plt.xlabel("Transaction Type")
plt.ylabel("Count")

# Plot violin plot of transaction amounts by type in client_cash_dataset
sns.violinplot(x='Type', y='Flow', data=client_cash_dataset)
plt.xlabel('Transaction Type')
plt.ylabel('Transaction Amount')
plt.title('Transaction Amount by Type')
plt.show()

# Plot violin plot of corn prices in Corn Price History dataset
sns.violinplot(x='Last', data=corn_dataset)
plt.xlabel('Corn Prices')
plt.title('Violin Plot of Corn Prices')
plt.show()

# Plot boxplot of corn prices in Corn Price History dataset
sns.boxplot(x='Last', data=corn_dataset)
plt.xlabel('Corn Prices')
plt.title('Boxplot of Corn Prices')
plt.show()

# Plot violin plot of balance in Client Cash Accounts dataset
sns.violinplot(x='Balance', data=client_cash_dataset)
plt.xlabel('Balance')
plt.title('Violin Plot of Balance')
plt.show()

# Plot boxplot of balance in Client Cash Accounts dataset
sns.boxplot(x='Balance', data=client_cash_dataset)
plt.xlabel('Balance')
plt.title('Boxplot of Balance')
plt.show()

# Perform time series decomposition on Flow column in client_cash_dataset
decomposition = seasonal_decompose(client_cash_dataset['Flow'], model='additive', period=1)
fig = decomposition.plot()
plt.show()

# Perform time series decomposition on Wheat Price History dataset
decomposition = seasonal_decompose(wheat_dataset['Price'], model='additive', period=1)
fig = decomposition.plot()
plt.show()

# Perform time series decomposition on Weather dataset
decomposition = seasonal_decompose(weather_dataset['Temperature'], model='additive', period=1)
fig = decomposition.plot()
plt.show()

# Perform time series decomposition on Corn Price History dataset
decomposition = seasonal_decompose(corn_dataset['Last'], model='additive', period=1)
fig = decomposition.plot()
plt.show()

# Perform time series decomposition on Client Cash Accounts dataset
decomposition = seasonal_decompose(client_cash_dataset['Balance'], model='additive', period=1)
fig = decomposition.plot()
plt.show()

# Encode categorical columns in Client Cash Accounts dataset
le = LabelEncoder()
le.fit(client_cash_dataset['Type'])
client_cash_dataset['Type'] = le.fit_transform(client_cash_dataset['Type'])
categorical_columns = ['Customer / Supplier', 'Reference', 'Description']
client_cash_dataset = pd.get_dummies(data=client_cash_dataset, columns=categorical_columns, dtype=int)

# Drop unnecessary columns in Client Cash Accounts dataset
client_cash_dataset = client_cash_dataset.drop(['Transaction Date', 'Client'], axis=1)

# Split data into training and testing sets for Random Forest Regression model - Client Cash Accounts dataset
x = client_cash_dataset.drop('Balance', axis=1)
y = client_cash_dataset['Balance']
x_train_cc, x_test_cc, y_train_cc, y_test_cc = train_test_split(x, y, test_size=0.2)

# Train a Random Forest Regression model on Client Cash Accounts dataset
random_forest_model_cc = RandomForestRegressor()
random_forest_model_cc.fit(x_train_cc, y_train_cc)
random_forest_acc_cc = random_forest_model_cc.score(x_test_cc, y_test_cc)
print('The accuracy of the Random Forest Regression model for Client Cash Accounts dataset is', random_forest_acc_cc)

# Split data into training and testing sets for Random Forest Regression model - Wheat Price History dataset
x = wheat_dataset.drop('Last', axis=1)
y = wheat_dataset['Last']
x_train_wheat, x_test_wheat, y_train_wheat, y_test_wheat = train_test_split(x, y, test_size=0.2)

# Train a Random Forest Regression model on Wheat Price History dataset
random_forest_model_wheat = RandomForestRegressor()
random_forest_model_wheat.fit(x_train_wheat, y_train_wheat)
random_forest_acc_wheat = random_forest_model_wheat.score(x_test_wheat, y_test_wheat)
print('The accuracy of the Random Forest Regression model for Wheat Price History dataset is', random_forest_acc_wheat)

# Split data into training and testing sets for Random Forest Regression model - Weather dataset
x = weather_dataset.drop(['Precipitation', 'Temperature'], axis=1)
y = weather_dataset[['Precipitation', 'Temperature']]
x_train_weather, x_test_weather, y_train_weather, y_test_weather = train_test_split(x, y, test_size=0.2)

# Train a Random Forest Regression model on Weather dataset
random_forest_model_weather = RandomForestRegressor()
random_forest_model_weather.fit(x_train_weather, y_train_weather)
random_forest_acc_weather = random_forest_model_weather.score(x_test_weather, y_test_weather)
print('The accuracy of the Random Forest Regression model for Weather dataset is', random_forest_acc_weather)

# Split data into training and testing sets for Random Forest Regression model - Corn Price History dataset
x = corn_dataset.drop('Last', axis=1)
y = corn_dataset['Last']
x_train_corn, x_test_corn, y_train_corn, y_test_corn = train_test_split(x, y, test_size=0.2)

# Train a Random Forest Regression model on Corn Price History dataset
random_forest_model_corn = RandomForestRegressor()
random_forest_model_corn.fit(x_train_corn, y_train_corn)
random_forest_acc_corn = random_forest_model_corn.score(x_test_corn, y_test_corn)
print('The accuracy of the Random Forest Regression model for Corn Price History dataset is', random_forest_acc_corn)
