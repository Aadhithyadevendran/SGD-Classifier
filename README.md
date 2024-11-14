# EX-07 SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1: Start

STEP 2: Load the Iris dataset and create a Pandas DataFrame with features and target.

STEP 3: Split the dataset into features (X) and target (y).

STEP 4: Split the data into training and testing sets using train_test_split.

STEP 5: Initialize the SGDClassifier with default parameters.

STEP 6: Train the classifier using the training data.

STEP 7: Predict the target values for the testing set.

STEP 8: Calculate and display the model's accuracy and confusion matrix.

STEP 9: End 

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: AADHITHYA D
RegisterNumber:  21222220001

```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


iris=load_iris()


df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target


print(df.head())


X = df.drop('target',axis=1)
y=df['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train,y_train)

y_pred=sgd_clf.predict(X_test)


accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)

```
## Output:
![image](https://github.com/user-attachments/assets/83d8bf6a-50e8-4fa0-97fe-f1690d3d0830)

![image](https://github.com/user-attachments/assets/92ba3d88-f9d3-4af0-ad61-fc52f49e36f7)

![image](https://github.com/user-attachments/assets/821beb26-67fe-4ab6-a29a-f2f78afd5ac0)


![image](https://github.com/user-attachments/assets/0c6f903c-83a1-46eb-9ed3-8ae6a75db472)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
