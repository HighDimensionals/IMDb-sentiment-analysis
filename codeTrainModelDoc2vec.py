import gensim
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence as LS

# We require LabeledSentence objects for some reason
def add_labels(reviews, typ):
    res = []
    for i in xrange(0, len(reviews)):
        review = reviews[i]
        label = "%s_%s" % (typ, i)
        res.append(LS(review.split(), [label]))

    return res

# Load training data
print "Loading training data..."
examples = pd.read_csv("clean_training_data_50.tsv", header=0, delimiter="\t", quoting=3)
train_reviews = add_labels(examples["review"].values, "train")
train_labels = examples["sentiment"].values
# Load unlabeled data
print "Loading unlabeled data..."
examples = pd.read_csv("clean_unlabeled_data_50.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_reviews = add_labels(examples["review"].values, "unlabeled")
# Load test data
print "Loading test data..."
examples = pd.read_csv("clean_test_data_50.tsv", header=0, delimiter="\t", quoting=3)
test_reviews = add_labels(examples["review"].values, "test")

# instantiate DM and DBOW
size = 400
dm = Doc2Vec(min_count=1, window=8, size=size, sample=1e-3, negative=5, workers=2)
dbow = Doc2Vec(min_count=1, window=8, size=size, sample=1e-3, dm=0, workers=2)

# Add the vocabulary to the models
print "Building dm vocabulary..."
dm.build_vocab(np.concatenate((train_reviews, unlabeled_reviews, test_reviews)))
print "Building dbow vocabulary..."
dbow.build_vocab(np.concatenate((train_reviews, unlabeled_reviews, test_reviews)))

# Now, according to the paper [TODO], we need to train the model
# several times randomizing the data each time
all_reviews = np.concatenate((train_reviews, test_reviews, unlabeled_reviews))
for i in xrange(0, 10):
    print "Training models the %d time" % (i+1)
    perm = np.random.permutation(all_reviews.shape[0])
    dm.train(all_reviews[perm])
    dbow.train(all_reviews[perm])

# Persist the model, as it takes too long to train
print "Persisting models..."
dm.save("dm_model")
dbow.save("dbow_model")
