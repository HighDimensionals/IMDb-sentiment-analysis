import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class Word:
    def __init__(self, word):
        self.word = word
        self.score = 0

    def add_score(self, label):
        if label == 0:
            self.score -= 1
        else:
            self.score += 1

class Lexicon:
    def __init__(self):
        self.words = {}

    def build(self, examples, labels):
        for i in xrange(0, len(examples)):
            for token in examples[i].split(" "):
                word = self.words.setdefault(token, Word(token))
                word.add_score(labels[i])

    def get_features(self, example, n=1):
        avg = 0
        count = 0
        scores = []
        for word in example.split(" "):
            if word in self.words:
                score = self.words[word].score
                count += 1
                scores.append(score)
                avg += score

        while len(scores) < 2*n:
            scores.append(0)

        scores.sort()
        avg = avg / count
        return scores[0:n] + [avg] + scores[-(n+1):-1]

    def transform(self, examples, n=1):
        ans = []
        for example in examples:
            ans.append(self.get_features(example, n))

        return ans

print "Loading training examples..."
examples = pd.read_csv("clean_training_data_50.tsv", header=0, delimiter="\t", quoting=3)
print "Creating lexicon..."
lexicon = Lexicon()
lexicon.build(examples["review"].values, examples["sentiment"].values)


# Main code, 50% of words, 75 min, 75 max of words
print "Generating training features..."
vectors = lexicon.transform(examples["review"].values, 75)
print "Training forest..."
forest = RandomForestClassifier(n_estimators = 150)
forest = forest.fit(vectors, examples["sentiment"].values)
print "Loading test examples..."
test = pd.read_csv("clean_test_data_50.tsv", header=0, delimiter="\t", quoting=3)
print "Generating test features..."
test_vectors = lexicon.transform(test["review"].values, 75)
print "Classifying..."
results = forest.predict(test_vectors)
output = pd.DataFrame( data={"id":test["id"], "sentiment":results} )
output.to_csv( "lexicon.csv", index=False, quoting=3 )
