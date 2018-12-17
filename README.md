# IMDb-sentiment-analysis

## Documentation

The following are instructions on how to run the different files.

Setup
------------------
All the data should be in the same folder as the python and matlab codes in order to run.

Requirements
------------------
For python, the following packages should be installed
- numpy
- scipy
- sckikit-learn
- gensim
- nltk
- pandas

Preprocessing
------------------
First step is to clean the data. For that purpose we have the clean_data.py script. It should be run as
- python clean_data.py
This will generate three files clean_training_data.tsv, clean_unlabeled_data.tsv, clean_test_data.tsv.
This could take several minutes.

Then we need to generate a list of words sorted by mutual information. For that we run the mutual_information.py script as
- python mutual_information.py > sorted_words

Finally we need to clean up the reviews even more, for that we run the script selector.py as
- python selector.py

Which will generate two new datasets of each type (train, unlabeled, test), one with only the top 50% and one for the top 40%.

-----------------
Algorithms
-----------------

Bayes
----------------
To run naive Bayes use the script naive_bayes.py and run it as
- python naive_bayes.py
It would generate a file named bayes.csv with the results.

Lexicon
----------------
To run the Lexicon classifier use the script lexicon.py and run it as
- python lexicon.py
It would generate a file named lexicon.csv with the results.

TF_IDF classifier
----------------
To run the tf-idf classifier use the script tfidf.py and run it as
- python tfidf.py
It would generate a file named tfidf.csv with the results.

doc2vec
----------------
This algorithm is run in two steps. We first train the distributed representation model and then we train a logistic regression algorithm on top of it.
For learning the distributed representation, run the file codeTrainModelDoc2vec.py as
- python codeTrainModelDoc2vec.py
This will generate several huge files containing the model. Also this could take hours to run.

To generate doc2vec features from the trained model, and run the classifier, 
the file codeGenFeaturesDoc2vec.py is used as
- python codeGenFeaturesDoc2vec.py
This will generate a file called deep.csv with the results.

word2vec
----------------
The word2vec algorithm is used to learn a vector representation for the vocabulary derived from the training dataset of 25000 labeled movie reviews. The word vectors pertaining to
the words in a review are averaged to obtain feature representation for each review. 
Specifically, only the adjectives, adverbs and verbs in a review are used in the averaging processing, since the adjectives, adverbs and verbs are more reflective of the sentiments
in a review.

For learning the word2vec representation using adjectives, adverbs and verbs only, run the file codeGenFeaturesAdjAdvVrb.py as
- python codeGenFeaturesAdjAdvVrb.py

The sentiment estimation of each review in the test dataset will be stored in "Word2Vec_AverageVectors_AdjAdvVrb.csv".



