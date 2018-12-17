from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import cross_validation
import pandas as pd
import numpy as np

print "Loading top 50% of the words..."
examples = pd.read_csv("clean_training_data_50.tsv", header=0, delimiter="\t", quoting=3)
reviews = examples["review"].values
labels = examples["sentiment"].values
n = reviews.size

print "Loading test examples..."
test = pd.read_csv("clean_test_data_50.tsv", header=0, delimiter="\t", quoting=3)

print "Loading unlabeled examples..."
unlabeled = pd.read_csv("clean_unlabeled_data_50.tsv", header=0, delimiter="\t", quoting=3)["review"].values

print 'Applyinf tf-if count...'
tfv = TfidfVectorizer(min_df=3,  max_features=5000, analyzer='word',token_pattern=r'\w{1,}',
                      ngram_range=(1, 1), use_idf=1,smooth_idf=1,sublinear_tf=1, stop_words = 'english')

all_data = np.concatenate((reviews, test["review"]))
tfv.fit(all_data)
features = tfv.transform(reviews)

logistic = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
# logistic = SGDClassifier(loss='log', penalty='l1')

print "Training logistic regression..."
logistic.fit(features, labels)
print "Cross validating..."
print "20 Fold CV Score: %.5f" % np.mean(cross_validation.cross_val_score(logistic, features, labels, cv=20, scoring='roc_auc'))
print "Transforming test data..."
test_vectors = tfv.transform(test["review"].values)
print "Predicting..."
results = logistic.predict_proba(test_vectors)[:,1]
output = pd.DataFrame( data={"id":test["id"], "sentiment":results} )
output.to_csv( "tfidf.csv", index=False, quoting=3 )
