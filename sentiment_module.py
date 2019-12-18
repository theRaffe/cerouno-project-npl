import pandas as pd
import re
from nltk import word_tokenize
from nltk.classify import NaiveBayesClassifier
import nltk.classify.util
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')


df_reviews = pd.read_csv('amazon-cell-phones-reviews/20190928-reviews.csv')
df_reviews.loc[df_reviews['rating'] <= 3, 'liked'] = False
df_reviews.loc[df_reviews['rating'] > 3, 'liked'] = True
stop_words = set(stopwords.words('english'))

threshold_factor = 0.7
model_index = int(threshold_factor * len(df_reviews))
print(f'model_index:{model_index}')
df_model_data = df_reviews.iloc[:model_index]

threshold_factor = 0.8

positive_reviews = df_model_data[df_model_data['liked'] == True]
negative_reviews = df_model_data[df_model_data['liked'] == False]

m_str = 'this is a sentence.and this other 12.90'
re.sub("[^\w]", " ",  m_str).split()
def get_list_words(reviews_str):
    token = str(reviews_str)
    return re.sub("[^\w]", " ",  token).split()

def extract_features(word_list):
    return dict([(word, True) for word in word_list if word.lower() not in stop_words])
#if word not in stop_words

positive_series = [get_list_words(review) for review in positive_reviews['body'].values]
negative_series = [get_list_words(review) for review in negative_reviews['body'].values]

positive_features = [(extract_features(a_review), 'Positive') for a_review in positive_series]
negative_features = [(extract_features(a_review), 'Negative') for a_review in negative_series]

print(len(positive_features))
print(len(negative_features))

# Split the data into train and test (80/20)
threshold_factor = 0.8
threshold_positive = int(threshold_factor * len(positive_features))
threshold_negative = int(threshold_factor * len(negative_features))

features_train = positive_features[:threshold_positive] + negative_features[:threshold_negative]
features_test = positive_features[threshold_positive:] + negative_features[threshold_negative:]  
print("\nNumber of training datapoints:", len(features_train))
print("Number of test datapoints:", len(features_test))

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(features_train)
print("\nAccuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test))

classifier.show_most_informative_features(15)

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, features_test))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(features_train)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, features_test))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(features_train)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, features_test))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(features_train)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, features_test))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(features_train)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, features_test))*100)

#SVC_classifier = SklearnClassifier(SVC())
#SVC_classifier.train(features_train)
#print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, features_test))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(features_train)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, features_test))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(features_train)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, features_test))*100)