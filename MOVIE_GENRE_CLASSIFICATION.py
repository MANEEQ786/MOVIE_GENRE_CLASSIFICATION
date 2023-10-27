#Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

pd.set_option('display.max_columns', None)

train_data = pd.read_csv('train_data.txt', sep=' ::: ', engine='python', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
test_data = pd.read_csv('test_data.txt', sep=' ::: ', engine='python', names=['ID', 'TITLE', 'DESCRIPTION'])
print(train_data.head())

#Remove missing values
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# Shuffle the training data
train_data = train_data.sample(frac=1, random_state=42)

# Split the training data into features (X) and labels (y)
X_train = train_data['DESCRIPTION']
y_train = train_data['GENRE']

X_test = test_data['DESCRIPTION']

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Choose a Classifier and Train the Model
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Make Predictions
y_pred = classifier.predict(X_test_tfidf)

# Create a DataFrame for the test predictions
test_predictions = pd.DataFrame({'ID': test_data['ID'], 'GENRE': y_pred})

# Save the predictions to a CSV file
test_predictions.to_csv('test_predictions.csv', index=False)

# Evaluate the model
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Then, train and evaluate the model on X_val and y_val
accuracy = accuracy_score(y_val, classifier.predict(tfidf_vectorizer.transform(X_val)))*100
report = classification_report(y_val, classifier.predict(tfidf_vectorizer.transform(X_val)), zero_division=0)

print("\n------------------------------------------------------")
print("------------------------------------------------------")

print(report)

print("------------------------------------------------------")
print(f'Accuracy: {accuracy:.2f}%')
print("------------------------------------------------------")



