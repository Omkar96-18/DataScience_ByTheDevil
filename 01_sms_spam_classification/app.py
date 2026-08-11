import streamlit as st
import pickle as pkl
import string
from nltk.corpus import stopwords
import nltk 
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pkl.load(open('vectorizer.pkl', 'rb'))
model = pkl.load(open('model.pkl', 'rb'))

st.title("SMS Spam/Ham Classifier")

input_text = st.text_area("Enter the text -> ")

if st.button('Predict'):
    transform_sms = transform_text(input_text)

    vector_input = tfidf.transform([transform_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else: 
        st.header("Not Spam")