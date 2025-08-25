import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# load dataset
dataset = pd.read_csv("dataset/Advertise.csv")
X = dataset.iloc[:, [0, 1, 2]].values
y = dataset.iloc[:, 3].values

# look at our dataset
print(dataset.head())
print(f'Info of X: {X}')
print(f'Info of y: {y}')

# split the dataset
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature scaling
sc = StandardScaler()
X_train_scaling = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# train the model
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)

# print out the model summary
print(dt.summary())

# make prediction
y_pred = dt.predict(X_test)

