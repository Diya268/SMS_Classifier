import streamlit as st
import pickle

#lets load the saved vectorizer and naive model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#sample text data and corresponding labels (replace with your actual data)
X_train = ["Sample text 1", "Sample text 2", "Sample text 3"]
y_train = [0, 1, 0] #Example labels (0 for negative, 1 for positive)

#create and train the TF-IDF vectorizer
tfidf = TfidfVectorizer(lowercase=True, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)

#create and train the Naive Bayes Classifier
mnb = MultinomialNB()
mnb.fit(X_train_tfidf, y_train)

# Save the trained TF-IDF vectorizer and Naive Bayes model to files
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(mnb, model_file)

#transform_text function for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)


#remove special characters and retaining alphanumeric words
    text = [word for word in text if word.isalnum]

#removing stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

#applying steming
    text = [ps.stem(word) for word in text]

    return " ".join(text)


#saving streamlit code
st.title("Email Spam Classifier")
input_sms = st.text_area("Enter message")
 
if st.button("Predict"):
    
    #preprocess
    transformed_sms = transform_text(input_sms)
    #vectorize
    vector_input = tfidf.transform([transformed_sms])
    #predict
    result = model.predict(vector_input)[0]
    #display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")  