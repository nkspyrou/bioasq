import sklearn
import pandas as pd
import nltk 
from sklearn import preprocessing, svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
import numpy as np
from scipy import stats

#nltk.download() 
#https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/


#1. PARSING
#parse dataset. Dataset has the format 'label', 'message', 'q_code' 
#1 if question belongs to category, 0 if it does not belong. SPARSE MATRIX
df = pd.read_csv('C:\\Users\\user\\Desktop\\annotated_questions\\h_pico.txt',       #foreground_background_0_1_pandata.txt', 
                   sep='\t', 
                   header=None,
                   names=['label','message', 'code'])


#df['label'] = df.label.map({'background': 0, 'foreground': 1})  

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
k=1 #initialize counter for k-fold cross-val
dummy_values = list()
bayes_values = list()
rforest_values = list()
svm_values = list()
nn_values = list()

for train_index, test_index in skf.split(counts , df['label']):
	X_train, X_test = counts[train_index], counts[test_index]
	y_train, y_test = df['label'][train_index], df['label'][test_index]
	
	
	print( "###############################################\n" + str(k) + " run stratified cross validation metrics:")	
	#dummy classifier
	dummy = DummyClassifier(strategy='stratified', random_state=0)
	dummy_model = dummy.fit(X_train, y_train)
	DummyClassifier(constant=None, random_state=0, strategy='stratified')
	dummy_predicted = dummy_model.predict(X_test)
	dummy_values.append(f1_score(y_test , dummy_predicted))
	print("Dummy F1:" + str(f1_score(y_test , dummy_predicted)))
	print(confusion_matrix(y_test, dummy_predicted)) 
	
	
	#naive bayes
	model = MultinomialNB().fit(X_train, y_train)  
	predicted = model.predict(X_test) 
	bayes_values.append(f1_score(y_test , predicted))
	print("NB F1:" + str(f1_score(y_test ,predicted)))
	print(confusion_matrix(y_test, predicted)) 
	
	
	#random forest
	clf = RandomForestClassifier(n_estimators=100, max_depth=40, random_state=0)
	model = clf.fit(X_train, y_train) 						  
	predicted = model.predict(X_test)
	rforest_values.append(f1_score(y_test , predicted))
	print("RF F1:" + str(f1_score(y_test , predicted)))
	print(confusion_matrix(y_test, predicted)) 
	
	
	#svm
	clf = svm.SVC(gamma='scale')
	model = clf.fit(X_train, y_train) 
	predicted = model.predict(X_test)
	svm_values.append(f1_score(y_test , predicted))
	print("SVM F1:" + str(f1_score(y_test , predicted)))
	print(confusion_matrix(y_test, predicted)) 
	
	
	#neural network
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1) #5,50  10,10 10,100
	model = clf.fit(X_train, y_train) 
	predicted = model.predict(X_test)
	nn_values.append(f1_score(y_test , predicted))
	print("NN F1:" + str(f1_score(y_test , predicted)))
	print(confusion_matrix(y_test, predicted)) 
	
	
	#update countare for k-fold
	k+=1
	
	



	
#STATS
#shapiro wilk stats.shapiro(df['bp_difference'])
#ttest stats.ttest_rel(df['bp_before'], df['bp_after'])
list_of_lists_of_f1 = (dummy_values , rforest_values , bayes_values , svm_values , nn_values )
list_of_names_of_lists_of_f1 = ("dummy_values" , "rforest_values" , "bayes_values" , "svm_values" , "nn_values")
shapiro_list = list()
man_witney_list = list()
ttest_list = list()
n = 0


#list of pairs of lists of F1 scores for identification of statistically significant differences between models
list_of_pairs = [(dummy_values, bayes_values), (rforest_values, dummy_values) , (rforest_values, bayes_values) , (rforest_values, svm_values) , (rforest_values, nn_values) , 
                  (svm_values, dummy_values), (svm_values, bayes_values) , (svm_values, nn_values), (nn_values, dummy_values), (nn_values, bayes_values),
				]		

#header for stats csv output				
list_of_pairnames = ["dummy_values, bayes_values", "rforest_values, dummy_values" , "rforest_values, bayes_values" , "rforest_values, svm_values" , "rforest_values, nn_values" , 
                  "svm_values, dummy_values", "svm_values, bayes_values" , "svm_values, nn_values", "nn_values, dummy_values", "nn_values, bayes_values",
				]
				
				
#t-test to check for differences in test performances				  
for pair in list_of_pairs:
	#ttest_list.append(str("{0:.3f}".format(stats.ttest_ind(pair[0],pair[1])[1])))
	man_witney_list.append(str("{0:.3f}".format(stats.ttest_ind(pair[0],pair[1])[1])))


#OUTPUT results in a txt(tsv) file for input in excel sheet	
fh = open("res.txt","w")
leaf= open("leaf.txt","w")#overleaf input
fh.write("F1 scores(over k-folds) : mean ± std\n")
head_f1 = "\t".join(list_of_names_of_lists_of_f1)
fh.write(head_f1)
fh.write("\n")

#output.f1_mean_std
for l in list_of_lists_of_f1:
	fh.write(str("{0:.2f}".format(np.mean(l)))+ "±" + str("{0:.2f}".format(np.std(l))) + "\t")
	leaf.write(str("{0:.2f}".format(np.mean(l)))+ "\pm" + str("{0:.2f}".format(np.std(l))) + "&")
fh.write("\n\n")

#output.normality
fh.write("Normality(Shapiro p-values):\n")
fh.write(head_f1)
fh.write("\n")
for l in list_of_lists_of_f1:
	fh.write(str("{0:.2f}".format(stats.shapiro(l)[1]))+ "\t")
fh.write("\n\n")

#output.ttests
fh.write("Statistical significance of pair difference (Wilcoxon-Mann-Whitney or ttest based on Shapiro result)\n")
head_stats = "\t".join(list_of_pairnames)
stats = "\t".join(man_witney_list)
fh.write(head_stats)
fh.write("\n")
fh.write(stats)

fh.close()


leaf.write(str("{0:.2f}".format(np.mean(l)))+ "\pm" + str("{0:.2f}".format(np.std(l))) + "&")
leaf.write("\n")
leaf.write("&".join(man_witney_list))
leaf.close()