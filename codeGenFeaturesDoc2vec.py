import gc
import gensim
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence as LS
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix, hstack

# We require LabeledSentence objects for some reason
def add_labels(reviews, typ):
    res = []
    for i in xrange(0, len(reviews)):
        review = reviews[i]
        label = "%s_%s" % (typ, i)
        res.append(LS(review.split(), [label]))

    return res

def gen_vectors(dm, dbow, reviews, size):
    dm_vectors =   np.concatenate([np.array(dm[review.labels[0]]).reshape(1, size) for review in reviews])
    dbow_vectors = np.concatenate([np.array(dbow[review.labels[0]]).reshape(1, size) for review in reviews])
    return np.hstack((dm_vectors, dbow_vectors))

# Load training data
print "Loading training data..."
examples = pd.read_csv("clean_training_data_50.tsv", header=0, delimiter="\t", quoting=3)
tr_reviews = examples["review"].values
train_reviews = add_labels(tr_reviews, "train")
train_labels = examples["sentiment"].values
# Load unlabeled data
print "Loading unlabeled data..."
examples = pd.read_csv("clean_unlabeled_data_50.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_reviews = add_labels(examples["review"].values, "unlabeled")
# Load test data
print "Loading test data..."
examples = pd.read_csv("clean_test_data_50.tsv", header=0, delimiter="\t", quoting=3)
ts_reviews = examples["review"].values
test_reviews = add_labels(ts_reviews, "test")
# Load the models
print "Loading models..."
size = 400
dm = Doc2Vec.load("dm_model")
dbow = Doc2Vec.load("dbow_model")
# Now get the vectors from the models
print "Generating vectors..."
# We are using memoery heavy operations, make sure we free all garbage before proceeding
gc.collect()
train_vectors     = gen_vectors(dm, dbow, train_reviews, size)
gc.collect()
test_vectors      = gen_vectors(dm, dbow, test_reviews, size)
# Free the huge memory taken by these models
dm = None
dbw = None
gc.collect()

# Generate tf-idf features
print "Generating tf-idf matrix"
tfv = TfidfVectorizer(min_df=3,  max_features=35000, analyzer='word',token_pattern=r'\w{1,}',
                      ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1, stop_words = 'english')
all_data = np.concatenate((tr_reviews, ts_reviews))
tfv.fit(all_data)
tf_train = tfv.transform(tr_reviews)
tf_test = tfv.transform(ts_reviews)
train_vectors = hstack([coo_matrix(train_vectors), tf_train])
test_vectors  = hstack([coo_matrix(test_vectors) , tf_test])

# Free more memory
tf_train = None
tf_test = None
tr_reviews = None
ts_reviews = None
tfv = None
gc.collect()

# Train a logistic regressor
print "Training logistic regression..."
logistic = LogisticRegression(penalty='l2', dual=True, tol=0.0001,class_weight=None, random_state=None)
logistic.fit(train_vectors, train_labels)
print "10 Fold CV Score: %.5f" % np.mean(cross_validation.cross_val_score(logistic, train_vectors, train_labels, cv=10, scoring='roc_auc'))
# Predict labels
print "Predicting..."
results = logistic.predict_proba(test_vectors)[:,1]
output = pd.DataFrame( data={"id":examples["id"], "sentiment":results} )
output.to_csv( "deep.csv", index=False, quoting=3 )
