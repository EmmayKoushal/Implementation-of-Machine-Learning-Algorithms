import pandas
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pandas.read_csv('Titanic.csv')

print(data.head())
y = data[['Survived']]
X = data[['Pclass','Age','SibSp','Parch','Fare']]
X.Age = X.Age.fillna(X.Age.mean())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.score(y_test, y_pred))