import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

#Data Collection & Processing

# loading the data from csv file to a Pandas DataFrame
calories = pd.read_csv('/content/calories.csv')

# print the first 5 rows of the dataframe
calories.head()

exercise_data = pd.read_csv('/content/exercise.csv')
exercise_data.head()

#Combining the two Dataframes
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)
calories_data.head()

# checking the number of rows and columns
calories_data.shape

# getting some informations about the data
calories_data.info()

# checking for missing values
calories_data.isnull().sum()

# get some statistical measures about the data
calories_data.describe()

#Data Visualization
sns.set()

# plotting the gender column in count plot
sns.countplot(calories_data['Gender'])

# finding the distribution of "Age" column
sns.distplot(calories_data['Age'])

# finding the distribution of "Height" column
sns.distplot(calories_data['Height'])

# finding the distribution of "Weight" column
sns.distplot(calories_data['Weight'])

#Finding the Correlation in the dataset

#Positive Correlation
#Negative Correlation
correlation = calories_data.corr()

# constructing a heatmap to understand the correlation

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)
calories_data.head()

#Separating features and Target
X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']
print(X)
print(Y)

#Splitting the data into training data and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# loading the model
model = XGBRegressor()

# training the model with X_train
model.fit(X_train, Y_train)

#Prediction on Test Data
test_data_prediction = model.predict(X_test)
print(test_data_prediction)

#Mean Absolute Error
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("Mean Absolute Error = ", mae)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy on test data = ', test_data_accuracy)

input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')


