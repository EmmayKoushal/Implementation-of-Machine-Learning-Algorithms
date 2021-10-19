import numpy
import pandas
from pandas.io.parsers import TextParser
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data = pandas.read_csv('car data.csv')
print(data.head())
X = data.iloc[:,:-1]
y = data.iloc[:,3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
r = r2_score(y_test, y_pred)
print(r)
