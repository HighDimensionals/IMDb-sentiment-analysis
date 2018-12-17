from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

class Ngram:
    def __init__(self, _ngram):
        self.ngram = _ngram
        self.positive = 1
        self.negative = 1

    def add_example(self, label):
        if(label == 0):
            self.negative += 1
        else:
            self.positive += 1

class BayesClassifier:
    def __init__(self):
        self.positive = 0
        self.negative = 0
        self.ngrams = {}

    def train_model(self, examples, labels, indices, n=1):
        for i in indices:
            words = examples[i].split(" ")
            items = []
            for j in xrange(0, len(words)-n+1):
                tmp = ""
                for k in xrange(0,n):
                    tmp += words[j+k]
                    if k < n-1:
                        tmp += " "
                items.append(tmp)

            for word in items:
                ngram = self.ngrams.setdefault(word, Ngram(word))
                ngram.add_example(labels[i])
                if(labels[i] == 0):
                    self.negative += 1
                else:
                    self.positive += 1

    def predict(self, example, treshold, n=1):
        ratio = self.negative * 1.0 / self.positive
        decision = (1.0/ratio)

        words = example.split(" ")
        items = []
        for j in xrange(0, len(words)-n+1):
            tmp = ""
            for k in xrange(0,n):
                tmp += words[j+k]
                if k < n-1:
                    tmp += " "
            items.append(tmp)

        for word in items:
            # Ignore things we have not seen (as we do not know how to treat them yet)
            if word in self.ngrams:
                ngram = self.ngrams[word]
                # Make sure the data is somewhat significant
                if ngram.positive > treshold or ngram.negative > treshold:
                    decision *= (ngram.positive * 1.0 / ngram.negative) * ratio

        if decision >= 1:
            return 1
        else:
            return 0

# Read the data
print "Loading training examples..."
examples = pd.read_csv("clean_training_data_50.tsv", header=0, delimiter="\t", quoting=3)

reviews = examples["review"].values
labels = examples["sentiment"].values
n = reviews.size

print "Training model..."
bayes = BayesClassifier()
bayes.train_model(reviews, labels, xrange(0, len(examples)), 2)

print "Loading test examples..."
test = pd.read_csv("clean_test_data_50.tsv", header=0, delimiter="\t", quoting=3)
print "Classifying..."
results = []
for t in test["review"]:
    pred = bayes.predict(t, 0, 2)
    results.append(pred)

output = pd.DataFrame( data={"id":test["id"], "sentiment":results} )
output.to_csv( "bayes.csv", index=False, quoting=3 )
