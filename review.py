import pandas as pd
import numpy as np
df = pd.read_csv('https://raw.githubusercontent.com/sharibhumayun/practice/main/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

import re
import nltk
nltk.download('stopwords')
import re
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review) 

from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer()
X= cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state =0)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train,y_train)

y_pred = model.predict(X_test)



import streamlit as st
import numpy as np


def Sentiment analysis(X_test):
   
    Input_values = np.asarray(X_test)
    Input_values = Input_values.reshape(1,-1)
    y_pred1 = model.predict(X_test)
    y_pred1
    
    
def main():
    st.title('**Restaurant review**')
    
    
  
   
    Review = st.text_input('Review')
    
    
    
    y_pred1 = ''
    
    if st.button('Restaurant Review'):
        y_pred1= Restaurant Review([Review])
        
    st.success(y_pred1)
        
        
        
if __name__=='__main__':
    main() 
