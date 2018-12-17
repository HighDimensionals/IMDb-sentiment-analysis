import math
import operator
import pandas as pd

class Word:
    def __init__(self):
        self.positive = 0
        self.negative = 0

    def add_example(self, label):
        if(label == 0):
            self.negative += 1
        else:
            self.positive += 1

    def conditional_prob(self, x, y, total_pos, total_neg):
        base = 0
        if y == 0:
            base = 1.0*self.negative / total_neg
        else:
            base = 1.0*self.positive / total_pos
        if x == 0:
            return 1.0 - base
        else:
            return base


    def prob(self, x, total):
        base = 1.0*(self.negative + self.positive) / total
        if x == 0:
            return 1.0 - base
        else:
            return base

class MutualInformation:
    def __init__(self):
        self.pos_reviews = 0
        self.neg_reviews = 0
        self.positive = 0
        self.negative = 0
        self.words = {}

    def add_examples(self, examples, labels, n=1):
        for i in xrange(0, len(examples)):
            words = examples[i].split(" ")
            tokens = []
            for j in xrange(0, len(words)-n+1):
                tmp = ""
                for k in xrange(0,n):
                    tmp += words[j+k]
                    if k < n-1:
                        tmp += " "
                tokens.append(tmp)

            for token in tokens:
                word = self.words.setdefault(token, Word())
                word.add_example(labels[i])
                if(labels[i] == 0):
                    self.negative += 1
                else:
                    self.positive += 1
            if(labels[i] == 0):
                self.neg_reviews += 1
            else:
                self.pos_reviews += 1

    def joint(self, x, y, word):
        return word.conditional_prob(x, y, self.positive, self.negative) * self.prob(y)

    def prob(self, y):
        if y == 0:
            return self.neg_prob
        else:
            return self.pos_prob

    def compute_information(self):
        self.total = (self.positive + self.negative)
        self.neg_prob = 1.0*(self.neg_reviews)/(self.neg_reviews + self.pos_reviews)
        self.pos_prob = 1.0*(self.pos_reviews)/(self.neg_reviews + self.pos_reviews)

        informations = []
        for key, word in self.words.iteritems():
            info = 0
            for y in [0,1]:
                for x in [0,1]:
                    pxy = self.joint(x, y, word)
                    if pxy > 0:
                        py = self.prob(y)
                        px = word.prob(x, self.total)
                        info += pxy*math.log(pxy/(px*py), 2)

            informations.append((key, info))

        return informations

print "Loading training examples..."
examples = pd.read_csv("clean_training_data.tsv", header=0, delimiter="\t", quoting=3)

mi = MutualInformation()
print "Parsing examples..."
mi.add_examples(examples["review"].values, examples["sentiment"].values, 1)
print "Computing mutual information..."
words = mi.compute_information()
print "Sorting by mutual info"
words.sort(key=operator.itemgetter(1), reverse=True)
for e in words:
    print e[0]
