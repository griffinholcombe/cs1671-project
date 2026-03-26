from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# note you need the data in a folder named as so for this to run
train = pd.read_csv('data/train.csv') # id,source,sub_source,lang,model,label,text,word_count,wc_bucket,key
dev = pd.read_csv('data/dev.csv') # id,source,sub_source,lang,model,label,text,word_count,wc_bucket,key

print("--------------------------------TRAIN DATASET INFO--------------------------------")
print(train.info())

print("--------------------------------TRAIN DATASET HEAD--------------------------------")
print(train.head())

print("----------------BEGIN CREATING UNIGRAM FEATURES----------------")
unigram_vectorizer = CountVectorizer()
unigram_vectorizer.fit(train['text'])
train_features = unigram_vectorizer.transform(train['text'])

print("----------------BEGIN BAG OF WORDS LOGISTIC REGRESSION----------------")
unigram_classifier = LogisticRegression(max_iter=1000)
unigram_classifier.fit(train_features, train['label'])
dev_gold_labels = dev['label']
dev_features = unigram_vectorizer.transform(dev['text'])
unigram_classifier_predictions = unigram_classifier.predict(dev_features) 

print(classification_report(dev_gold_labels, unigram_classifier_predictions))
