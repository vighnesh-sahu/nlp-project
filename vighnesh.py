import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
with open('rt-polarity.pos', 'r', encoding='latin-1') as f:
    positive_reviews = f.readlines()

with open('rt-polarity.neg', 'r', encoding='latin-1') as f:
    negative_reviews = f.readlines()
    
train_pos = positive_reviews[:4000]
train_neg = negative_reviews[:4000]
val_pos = positive_reviews[4000:4500]
val_neg = negative_reviews[4000:4500]
test_pos = positive_reviews[4500:]
test_neg = negative_reviews[4500:]

train_texts = train_pos + train_neg
train_labels = [1] * len(train_pos) + [0] * len(train_neg)
val_texts = val_pos + val_neg
val_labels = [1] * len(val_pos) + [0] * len(val_neg)
test_texts = test_pos + test_neg
test_labels = [1] * len(test_pos) + [0] * len(test_neg)

vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, train_labels)

val_predictions = rf.predict(X_val)
val_accuracy = accuracy_score(val_labels, val_predictions)

test_predictions = rf.predict(X_test)
test_accuracy = accuracy_score(test_labels, test_predictions)

conf_matrix = confusion_matrix(test_labels, test_predictions)

TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)  

classification_rep = classification_report(test_labels, test_predictions)

plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(f"TP: {TP}")
print(f"TN: {TN}")
print(f"FP: {FP}")
print(f"FN: {FN}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Confusion Matrix:\n {conf_matrix}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print("Classification Report:\n", classification_rep)

