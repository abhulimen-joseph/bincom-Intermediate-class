import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("Titanic.csv")


df = df[["survived", "sex", "age", "fare", "sibsp", "parch", "embarked"]]


# fills the stated column that has Nan with median and mode 
df.fillna({"age": df["age"].median()}, inplace= True)
df.fillna({"embarked": df["embarked"].mode()[0]}, inplace= True)

# Encode categorial column converts it into figures 
encoder = LabelEncoder()
df["sex"] = encoder.fit_transform(df["sex"])
df["embarked"] = encoder.fit_transform(df["embarked"])

# Split features and labels
x = df.drop("survived", axis=1)
y = df["survived"]

X_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42)

# train Logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# make prediction
y_pred = model.predict(x_test)

# evaluate performance 
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification:\n", classification_report(y_test, y_pred))

# interpretation
feature_importance = pd.DataFrame({
    "Features": x.columns,
    "Co_efficient": model.coef_[0]

}).sort_values(by="Co_efficient", ascending= False)

print(feature_importance)