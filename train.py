import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score


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
# Evaluation
print(" Attribute-wise Evaluation:\n")

results_lines = []
per_attr_accuracy = {}
per_attr_f1 = {}

for i, column in enumerate(y.columns):
    print(f"\n {column.upper()}:\n")
    report = classification_report(y_test.iloc[:, i], y_pred[:, i], zero_division=0)
    print(report)

    acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    f1 = f1_score(y_test.iloc[:, i], y_pred[:, i], average="macro", zero_division=0)
    per_attr_accuracy[column] = acc
    per_attr_f1[column] = f1

    results_lines.append(f"{column}: accuracy={acc:.3f}, macro_f1={f1:.3f}")

overall_f1 = sum(per_attr_f1.values()) / len(per_attr_f1)
overall_accuracy = sum(per_attr_accuracy.values()) / len(per_attr_accuracy)

print("\n===== SUMMARY =====")
for line in results_lines:
    print(line)
print(f"\nOverall average accuracy: {overall_accuracy:.3f}")
print(f"Overall macro-averaged F1 score: {overall_f1:.3f}")

# Save evaluation results to a file for documentation
with open("evaluation_results.txt", "w") as f:
    f.write("Attribute-wise Evaluation\n")
    f.write("=========================\n\n")
    for line in results_lines:
        f.write(line + "\n")
    f.write(f"\nOverall average accuracy: {overall_accuracy:.3f}\n")
    f.write(f"Overall macro-averaged F1 score: {overall_f1:.3f}\n")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model trained and saved")