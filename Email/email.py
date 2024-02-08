import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load your dataset (replace 'spam.csv' with your actual file name)
data = pd.read_csv('spam.csv', encoding='latin1')

# Prepare data for training
X = data['v2']  # Email text
y = data['v1']  # Labels (spam or not)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Training the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_tfidf, y)

# Function to interact with the model
def predict_spam_or_not():
    while True:
        # Ask for user input
        new_email_text = input("Enter the email text (or 'quit' to exit): ")

        if new_email_text.lower() == 'quit':
            break

        # Transform the input using the same vectorizer
        new_email_tfidf = tfidf_vectorizer.transform([new_email_text])

        # Predict if the input email is spam or not
        prediction = nb_classifier.predict(new_email_tfidf)

        # Print the prediction (1 for spam, 0 for not spam)
        if prediction[0] == 'spam':
            print("This email is predicted to be Spam.")
        else:
            print("This email is predicted to be Ham.")

# Call the function to start the interaction
predict_spam_or_not()
