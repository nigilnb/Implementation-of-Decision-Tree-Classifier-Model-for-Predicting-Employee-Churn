# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries required for data processing and machine learning.
2. Load and preprocess the dataset using Pandas.
3. Separate features (X) and target (y) variables.
4. Split the dataset into training and testing sets.
5. Train the model, predict results, and evaluate accuracy.
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NIGIL.S
RegisterNumber:212225240100
*/
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

data = pd.read_csv("C:/Users/acer/Downloads/Employee.csv")

print("Dataset Sample:")
print(data.head())

le = LabelEncoder()

data['Departments '] = le.fit_transform(data['Departments '])
data['salary'] = le.fit_transform(data['salary'])


X = data.drop("left", axis=1)
y = data["left"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = DecisionTreeClassifier()


model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Predicted Values:", y_pred)
print("Accuracy of the model:", accuracy)
print("\nClassification Report\n",classification_report(y_test,y_pred))

## Output:
<img width="684" height="394" alt="image" src="https://github.com/user-attachments/assets/29bd44b5-7109-4690-8749-ea31930822b4" />

<img width="225" height="67" alt="image" src="https://github.com/user-attachments/assets/fba39228-9520-4736-88fe-59a2c6791b87" />

<img width="471" height="230" alt="image" src="https://github.com/user-attachments/assets/5bb6dc9e-2ce7-4d17-a03f-a8b61d41c0c8" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
