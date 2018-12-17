import pandas as pd

def process(examples, vocabulary):
    for i in xrange(0, examples["review"].size):
        tokens = examples["review"][i].split(" ")
        words = [w for w in tokens if w in vocabulary]
        examples.set_value(i, "review", " ".join(words))

print "Reading words sorted by mutual information..."
words = [line.strip() for line in open('data/sorted_words')]

print "Generating new datasets..."
for p in xrange(40, 51, 10):
    print "Selecting top %d%%" % (p)
    print "Reading test data..."
    print "Processing train dataset..."
    examples = pd.read_csv("clean_training_data.tsv", header=0, delimiter="\t", quoting=3)
    limit = (len(words)*p) // 100
    restricted = set(words[0:limit])
    process(examples, restricted)
    examples.to_csv("clean_training_data_" + str(p) + ".tsv", sep="\t", index=False)
    print "Processing unlabeled dataset..."
    examples = pd.read_csv("clean_unlabeled_data.tsv", header=0, delimiter="\t", quoting=3)
    limit = (len(words)*p) // 100
    restricted = set(words[0:limit])
    process(examples, restricted)
    examples.to_csv("clean_unlabeled_data_" + str(p) + ".tsv", sep="\t", index=False)
    print "Processing test dataset..."
    examples = pd.read_csv("clean_test_data.tsv", header=0, delimiter="\t", quoting=3)
    limit = (len(words)*p) // 100
    restricted = set(words[0:limit])
    process(examples, restricted)
    examples.to_csv("clean_test_data_" + str(p) + ".tsv", sep="\t", index=False)

