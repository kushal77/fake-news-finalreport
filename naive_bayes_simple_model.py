import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


csv.field_size_limit(100000000)

# Load train data
train_file_path = "C:/project/train_data.csv"
selected_columns = ["type", "content"]

train_data_chunks = []
for chunk in pd.read_csv(train_file_path, usecols=selected_columns, chunksize=100000, on_bad_lines='warn', engine='python'):
    chunk = chunk.dropna()
    train_data_chunks.append(chunk)
train_data = pd.concat(train_data_chunks, ignore_index=True).sample(n=1000, random_state=42)

# Load test data
test_file_path = "C:/project/train_data.csv"
test_data_chunks = []
for chunk in pd.read_csv(test_file_path, usecols=selected_columns, chunksize=100000, on_bad_lines='warn', engine='python'):
    chunk = chunk.dropna()
    test_data_chunks.append(chunk)
test_data = pd.concat(test_data_chunks, ignore_index=True).sample(n=1000, random_state=42)


fake_labels = {'fake', 'satire', 'bias', 'conspiracy', 'state', 'junksci', 'hate', 'clickbait', 'unreliable'}
not_fake_labels = {'political', 'reliable'}

train_data['binary_label'] = train_data['type'].apply(lambda x: 1 if x in not_fake_labels else 0)
test_data['binary_label'] = test_data['type'].apply(lambda x: 1 if x in not_fake_labels else 0)

# Features and labels
X_train = train_data['content']
y_train = train_data['binary_label']
X_test = test_data['content']
y_test = test_data['binary_label']

# TF-IDF Vectorization 
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Naive Bayes model for classification
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
y_pred = nb_model.predict(X_test_vec)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print("Naive Bayes Classification Results:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
