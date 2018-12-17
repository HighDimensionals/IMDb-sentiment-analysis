from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import pandas as pd
import re
import nltk

def preprocess(examples):
    print "Cleaning %s examples" % examples["review"].size
    clean_examples = []
    negations = ["no", "not", "dont", "don't", "nor"]
    for i in xrange( 0, examples["review"].size):
        no_html = BeautifulSoup(examples["review"][i],'html.parser')
        letters_only = re.sub("[^a-zA-Z]", " ", no_html.get_text())
        tokens = letters_only.lower().split()
        # Add not to negations
        for j in xrange(0, len(tokens) - 1):
            if tokens[j+1] not in stopwords.words("english") and tokens[j] in negations:
                tokens[j+1] = "not_" + tokens[j+1]

        tokens = [w for w in tokens if not w in stopwords.words("english")]
        examples.set_value(i, "review", " ".join(tokens))
        if(i%250 == 0):
            print "%d reviews out of %d have been processed" % (i+1, examples["review"].size)
    return clean_examples

examples = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
preprocess(examples)
examples.to_csv("clean_training_data.tsv", sep="\t")

examples = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
preprocess(examples)
examples.to_csv("clean_unlabeled.tsv", sep="\t")

examples = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
preprocess(examples)
examples.to_csv("clean_test_data.tsv", sep="\t")
