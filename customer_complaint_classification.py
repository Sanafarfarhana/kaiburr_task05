# Consumer Complaint Classification

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('consumer_complaints.csv')  # Replace with actual file path

# Filter relevant categories
categories = {
    'Credit reporting, credit repair services, or other personal consumer reports': 0,
    'Debt collection': 1,
    'Consumer Loan': 2,
    'Mortgage': 3
}
df = df[df['Product'].isin(categories.keys())]
df = df[['Product', 'Consumer complaint narrative']].dropna()
df['label'] = df['Product'].map(categories)
df = df[['Consumer complaint narrative', 'label']].rename(columns={'Consumer complaint narrative': 'text'})

# Plot class distribution
sns.countplot(x='label', data=df)
plt.title("Distribution of Complaint Categories")
plt.show()

# Text preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'LightGBM': LGBMClassifier()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

# Show accuracy comparison
results_df = pd.DataFrame(results.items(), columns=['Model', 'Accuracy'])
print("\nModel Accuracy Comparison:\n", results_df.sort_values(by='Accuracy', ascending=False))

# Evaluate best model (Logistic Regression assumed here)
best_model = LogisticRegression(max_iter=1000)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Print classification report and plot confusion matrix
print("\nClassification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Predict function for new text
def predict_new(texts):
    clean = [preprocess(t) for t in texts]
    vect = tfidf.transform(clean)
    preds = best_model.predict(vect)
    reverse_map = {v: k for k, v in categories.items()}
    return [reverse_map[p] for p in preds]

# Example prediction
example_texts = [
    "They reported a wrong credit amount and did not fix it even after 3 months.",
    "They are calling me daily to repay a loan I never took."
]
print("\nPredictions:\n", predict_new(example_texts))
