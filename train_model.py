import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Step 1: Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
                 sep="\t", names=["label", "message"])

# Step 2: Convert labels to numbers
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label_num"], test_size=0.2, random_state=42)

# Step 4: Convert text to numbers
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Step 5: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Save model & vectorizer
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved!")
