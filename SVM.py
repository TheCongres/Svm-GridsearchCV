# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# read the dataset into a pandas dataframe
dt = r"C:\Users\LENOVO\Downloads\Compressed\FNews\bodyPerformance1.csv"
df = pd.read_csv(dt)

# print number of samples in the dataset 
print(len(df))

# print first lines of the dataset
print(df.head())

# print name of columns and thier types
print(df.dtypes)

# print dataset classes (target column values)
print(df.Blass.unique())

# check if the dataset is balanced
print(df.Blass.value_counts())

# check is the dataset contains missing values (NaN, valeurs manquantes)
print(df.isna().sum())


# split the dataset into features and target variable
X = df.drop(['Blass'], axis=1)
y = df['Blass']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define the SVM classifier
svm = SVC()

# define the parameter grid to search over
param_grid = {'C': [1,0.1, 0.01, 0.001 ],
              'gamma': [ 1,0.1, 0.01, 0.001 ],
              'kernel': ['rbf', 'sigmoid']}

# perform grid search to find the best parameters
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# print the best parameters found by grid search
print("Best parameters: ", grid_search.best_params_)

# use the best parameters to train the model
svm = SVC(C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'], kernel=grid_search.best_params_['kernel'])
svm.fit(X_train, y_train)

# make predictions on the test set
y_pred = svm.predict(X_test)

# calculate the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score: ", accuracy)

# print the classification report
report = classification_report(y_test, y_pred)
print("Classification report: \n", report)

# print the confusion matrix
matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix: \n", matrix)

import matplotlib.pyplot as plt

# plot the accuracy and iteration values
plt.plot(grid_search.cv_results_['mean_test_score'])
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('SVM Accuracy vs. Iteration')
plt.show()