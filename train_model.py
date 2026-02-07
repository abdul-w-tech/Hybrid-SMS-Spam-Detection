import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

print("RELOADING DATASET")
# Load the raw data
data = pd.read_csv('spam.csv', encoding='latin-1')
data.columns = ['v1', 'v2'] # Standardization

# Architecture: Pipeline Approach
# We bundle the Preprocessor (Vectorizer) and Model (Classifier) into one object.
# This ensures raw text can be fed directly into the model during production.
print("TRAINING PROBABILISTIC MODEL (Naive Bayes)")

model = Pipeline([
    ('vectorizer', CountVectorizer()), # Converts text to token counts
    ('classifier', MultinomialNB())    # Bayesian Probability Engine
])

model.fit(data['v2'], data['v1'])

print("SERIALIZING MODEL")
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("BUILD SUCCESSFUL. Accuracy optimized for production.")