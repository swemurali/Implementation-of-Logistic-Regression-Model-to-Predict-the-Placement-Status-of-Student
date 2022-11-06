# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries .
2. Load the dataset and check for null data values and duplicate data values in the dataframe.
3. Import label encoder from sklearn.preprocessing to encode the dataset.
4. Apply Logistic Regression on to the model.
5. Predict the y values.
6. Calculate the Accuracy,Confusion and Classsification report.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: M.Suwetha
RegisterNumber: 212221230112  
*/
import pandas as pd
df=pd.read_csv("Placement_Data(1).csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![m1](https://user-images.githubusercontent.com/94165336/200160091-dc97251b-b74f-4970-a8ea-1e953940a031.png)
![m2](https://user-images.githubusercontent.com/94165336/200160096-cc9b51f3-e227-4930-b71c-b1547c11811c.png)
![m3](https://user-images.githubusercontent.com/94165336/200160108-333a9911-8a91-4981-acf6-0f20f21e39be.png)
![m6](https://user-images.githubusercontent.com/94165336/200160168-bb943154-553a-4848-a5d4-f0721a398212.png)
![m7](https://user-images.githubusercontent.com/94165336/200160172-7fe008cf-e3ff-48e9-aa1b-539020c6eaae.png)
![m8](https://user-images.githubusercontent.com/94165336/200160178-3b678420-af21-4556-9725-33c26c2fd709.png)
![m9](https://user-images.githubusercontent.com/94165336/200160183-0b791d6f-d6dd-440b-93bd-91b2ae54eb8a.png)
![m10](https://user-images.githubusercontent.com/94165336/200160189-410692c0-967e-4d1c-a23a-a28e3184170b.png)
![m11](https://user-images.githubusercontent.com/94165336/200160192-4e3f746f-383c-4b96-a444-1309b38894f8.png)
![m12](https://user-images.githubusercontent.com/94165336/200160198-8a463f11-7e11-4d16-88e4-9e7cfd875fa3.png)
![m13](https://user-images.githubusercontent.com/94165336/200160201-4dc94d93-b96e-401f-9584-2bd0844172a9.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
