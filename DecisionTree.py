  import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

data = pandas.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/9_decision_tree/Exercise/titanic.csv')
print(data.head())
data = data[['Survived','Pclass','Sex','Age','Fare']]
print(data.head())

data['Age'] = data.groupby("Sex")['Age'].transform(lambda x: x.fillna(x.mean()))
le = LabelEncoder()
data['Sex'] = le.fit_transform(data.Sex)
data.head()

X = data[['Pclass','Sex','Age','Fare']]
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier(random_state=0, criterion='gini')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
plt.figure(figsize=(15,10))
tree.plot_tree(clf, filled=True)
plt.show()