import pandas as pd
from gensim.models import Word2Vec
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import nltk
import numpy as np

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Read data from files 
train = pd.read_csv( "labeledTrainData.tsv", header=0, 
                     delimiter="\t", quoting=3, encoding="utf-8" )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3, encoding="utf-8" )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, 
                               delimiter="\t", quoting=3, encoding="utf-8" )


model = Word2Vec.load("300features_40minwords_10context")

# Get list of words from a review
def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Average all the word vectors in the list of 'words'
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec

# creates a list of average feature vectors from a set of reviews.. an average vector for each review
def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%5000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs

# Average all the word vectors in the list of 'words'
def makeFeatureVec2(words, model, num_features,relevant_words,counter):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in relevant_words:
            if word in index2word_set: 
                nwords = nwords + 1.
                featureVec = np.add(featureVec,model[word])

	
	
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    if nwords == 0:
        print counter
    return featureVec


# creates a list of average feature vectors from a set of reviews.. an average vector for each review
def getAvgFeatureVecs2(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    reviewFeatureVecs=[]
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecsAdjAdvVrb = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%5000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       # 
       
       # Clean the review
       clean_review = review_to_wordlist( review, \
                           remove_stopwords=True )
       
       text_temp=review.strip().lower()
       abbrevs={'.':'. ','?':'? ','!':'! '}
       for abbrev in abbrevs:
           text_temp = text_temp.replace(abbrev,abbrevs[abbrev])       

       # Find the adjectives and adverbs from the original review
       raw_sentences = tokenizer.tokenize(text_temp)

       relevant_words_AdjAdvVrb=[]
       
       for raw_sentence in raw_sentences:
           mywords = nltk.tokenize.word_tokenize(raw_sentence)
           aa=nltk.pos_tag(mywords)
           for s in aa:
               if s[1] == 'JJ':
                   relevant_words_AdjAdvVrb.append(s[0])
                   
               if s[1] == 'RB':
                   relevant_words_AdjAdvVrb.append(s[0])
                   
               if s[1] == 'VB' or s[1] == 'VBD' or s[1] == 'VBG' or s[1] == 'VBN' or s[1] == 'VBP' or s[1] == 'VBZ':
                   relevant_words_AdjAdvVrb.append(s[0])
                   
       
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecsAdjAdvVrb[counter] = makeFeatureVec2(clean_review, model, \
           num_features,relevant_words_AdjAdvVrb,counter)
       
       #
       # Increment the counter
       counter = counter + 1.

    reviewFeatureVecs.append(reviewFeatureVecsAdjAdvVrb)
    
    return reviewFeatureVecs




num_features=300

print "Creating average feature vecs for test reviews"

testAdjAdvVrb = getAvgFeatureVecs2( test["review"], model, num_features )

# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

print "Fitting a random forest to labeled training data..."
forest = forest.fit( trainAdjAdvVrb, train["sentiment"] )

# Test & extract results 
result = forest.predict( testAdjAdvVrb )

# Write the test results 
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Word2Vec_AverageVectors_AdjAdvVrb.csv", index=False, quoting=3 )


