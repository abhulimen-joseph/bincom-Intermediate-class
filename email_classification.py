import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("spam_ham_dataset.csv")
print(list(df.columns))
# print(df.isna().sum())

#converted the label column to figures 
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])
# print(df["label"])

#feature and target 
x = df["text"]
y = df["label"]

#converting words to numerical value 
vectorizer = CountVectorizer(stop_words="english")
X_vect = vectorizer.fit_transform(x)

# Split into training and testing 
X_train, X_test, Y_train, Y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Logistic regression 
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

#make prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, y_pred))
print("\nClassification Report:\n", classification_report(Y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, y_pred))
