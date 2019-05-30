import sklearn
import pandas as pd
import nltk 
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
import numpy as np
#nltk.download() 
#https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/


#1. PARSING
#parse dataset. Dataset has the format 'label', 'message', 'q_code' 
#1 if question belongs to category, 0 if it does not belong. SPARSE MATRIX
df = pd.read_csv('C:\\Users\\user\\Desktop\\newpicoannot.txt', 
                   sep='\t', 
                   header=None,
                   names=['label','message', 'code'])


#df['label'] = df.label.map({'0': 0, '1': 1})  

#2. PREPROCESSING
#everything to lowercase
df['message'] = df.message.map(lambda x: x.lower()) 

#removing punctuation
df['message'] = df.message.str.replace('[^\w\s]', '') 

df['message'] = df['message'].apply(nltk.word_tokenize)   

#word stemming
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
df['message'] = df['message'].apply(lambda x: [stemmer.stem(y) for y in x]) 

#count vectorizer

# This converts the list of words into space-separated strings
df['message'] = df['message'].apply(lambda x: ' '.join(x))
count_vect = CountVectorizer()  
counts = count_vect.fit_transform(df['message'])  

#Term Frequency Inverse Document Frequency
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer().fit(counts)
counts = transformer.transform(counts) 

#training and test sets
#b.stratified crossvalidation
skf = StratifiedKFold(n_splits=3)
skf.get_n_splits(counts , df['label'])
n=1
crossval_metrics_list = list()
for train_index, test_index in skf.split(counts , df['label']):
	X_train, X_test = counts[train_index], counts[test_index]
	y_train, y_test = df['label'][train_index], df['label'][test_index]
	#dummy classifier
	if n==1:
		dummy = DummyClassifier(strategy='most_frequent', random_state=0)
		dummy_model = dummy.fit(X_train, y_train)
		DummyClassifier(constant=None, random_state=0, strategy='most_frequent')
		dummy_predicted = dummy_model.predict(X_test)
		print("Dummy metrics:\n")
		print(dummy_model.score(X_test, y_test))
		print(confusion_matrix(y_test, dummy_predicted)) 
	#naive bayes
	print( "###############################################\n" + str(n) + " run stratified cross validation metrics:")
	model = MultinomialNB().fit(X_train, y_train)  
	predicted = model.predict(X_test)
	print("Naive bayes metrics:\n")
	print(np.mean(predicted == y_test)) 
	crossval_metrics_list.append(np.mean(predicted == y_test))
	print(confusion_matrix(y_test, predicted)) 
	n += 1

print("###############################################\nMean metrics of all cross val naive bayes rounds:\n" 
		+ str(np.mean(crossval_metrics_list)) + " +- " +
		str(np.std(crossval_metrics_list)))




 
