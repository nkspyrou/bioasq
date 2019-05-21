import sklearn
import pandas as pd
import nltk 
#nltk.download() 
#https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/


#1. PARSING
#parse dataset. Dataset has the format [label] \t [text] \n
df = pd.read_csv('C:\\Users\\user\\Desktop\\foreground_background_pandata.txt',  
                   sep='\t', 
                   header=None,
                   names=['label', 'message'])

				   
				   #change labels to numbers				   
df['label'] = df.label.map({'background': 0 , 'foreground': 1})


#2. PREPROCESSING
#everything to lowercase
#df['message'] = df.message.map(lambda x: x.lower()) 

#removing punctuation
df['message'] = df.message.str.replace('[^\w\s]', '') 

df['message'] = df['message'].apply(nltk.word_tokenize)   

#word stemming
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

df['message'] = df['message'].apply(lambda x: [stemmer.stem(y) for y in x]) 

#count vectorizer
from sklearn.feature_extraction.text import CountVectorizer

# This converts the list of words into space-separated strings
df['message'] = df['message'].apply(lambda x: ' '.join(x))

count_vect = CountVectorizer()  
counts = count_vect.fit_transform(df['message'])  

#Term Frequency Inverse Document Frequency
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts) 

#split data to training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.15, random_state=69)  


#3. MODEL
#training
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB().fit(X_train, y_train)  


#4. EVALUATION
#evaluation (test set)
import numpy as np

predicted = model.predict(X_test)

print(np.mean(predicted == y_test))  

#confusion matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predicted))  


 
