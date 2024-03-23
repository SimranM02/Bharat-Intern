import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

titanic_data = pd.read_csv("titanic.csv")
print(titanic_data.head())
print(titanic_data.info())
print(titanic_data.describe())

imputer = SimpleImputer(strategy="mean")
titanic_data["Age"] = imputer.fit_transform(titanic_data[["Age"]])

titanic_data = pd.get_dummies(titanic_data, columns=["Sex", "Embarked"], drop_first=True)

X = titanic_data.drop(["Survived", "PassengerId", "Name", "Ticket", "Cabin"], axis=1)
y = titanic_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

probabilities = model.predict_proba(X_test_scaled)[:, 1]

plt.figure(figsize=(10, 6))
plt.hist(probabilities, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Predicted Probability of Survival')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities of Survival')
plt.grid(True)
plt.show()
