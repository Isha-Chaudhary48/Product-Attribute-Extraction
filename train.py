import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Load dataset
df = pd.read_csv("dataset.csv")

# Clean
df = df.fillna("unknown")

# Split
X = df["description"]
y = df.drop(columns=["description"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Evaluation
print(" Attribute-wise Evaluation:\n")


for i, column in enumerate(y.columns):
    print(f"\n {column.upper()}:\n")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i], zero_division=0))

# Save
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model trained and saved")