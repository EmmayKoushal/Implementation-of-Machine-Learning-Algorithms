import pandas
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pandas.read_csv('train.csv')
print(data.head())
X = data[['Gender', 'Married', 'Dependents', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Property_Area']]
y = data[['Loan_Status']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = svm.LinearSVC(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))