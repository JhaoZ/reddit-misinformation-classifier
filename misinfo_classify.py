from inspect import Parameter
from random import seed
from re import search
from tkinter import Grid
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

pd.options.mode.chained_assignment = None  # default='warn'

#Loading in data and setting to DataFrame
data = pd.read_csv("TestingData.csv")
df = pd.DataFrame(data)
df.fillna('', inplace = True)
stop = ENGLISH_STOP_WORDS
print(stop)

X = df[['Title', 'Text']]
y = df['Target']
X = X.reset_index()

print(y.value_counts())

#Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

def combine_df(df1):
    df1['Title'] = df1['Title'].astype(str)
    df1['Text'] = df1['Text'].astype(str)
    df1['Title'] = df1['Title'].str.lower()
    df1['Text'] = df1['Text'].str.lower()
    df1['Title'] = df1['Title'].str.strip()
    df1['Text'] = df1['Text'].str.strip()
    df1['Text'] = df1['Text'].apply(lambda x: "{}{}".format(' ', x))
    df1['Text'] = df1['Text'].str.replace(r'[^\w\s]', '')
    df1['Title'] = df1['Title'].str.replace(r'[^\w\s]', '')
    return (df1['Title'] + df1['Text']).str.split(' ').apply(lambda x: ' '.join(k for k in x if k not in stop))

X_train['Post'] = combine_df(X_train)
print(X_train.head())

# Create bag of words
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train['Post'])
print(X_train_counts)

# Frequency normalizing
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf= tf_transformer.transform(X_train_counts)

clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, tol=None).fit(X_train_tf, y_train)
# clf = LogisticRegression().fit(X_train_tf, y_train)
# clf = BernoulliNB().fit(X_train_tf, y_train)
# s_clf = SGDClassifier(penalty='l2', max_iter = 3, tol = None)
# parameters = {'loss': ('hinge', 'log'), 'alpha': (13-4, 1e-3, 1e-2)}
# clf = GridSearchCV(s_clf, parameters).fit(X_train_tf, y_train)
#Testing
docs_new = []
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    res = False
    if category > 0.5:
        res = True
    print('%r => %s' % (doc, res))

searchList = X_test['Title'].astype(str) + X_test['Text'].astype(str)
searchList = searchList.reset_index()
X_test['Post'] = combine_df(X_test)
X_new_counts = count_vect.transform(X_test['Post'])
X_new_tfidf = tf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

#print(predicted)
#print(y_test)

print("Metrics:")
print("Accuracy Score:", accuracy_score(predicted, y_test))
print("F1 score:", f1_score(predicted, y_test))
print("Confusion Matrix:")
matrix = confusion_matrix(predicted, y_test)
print(matrix)
print("True Negatives:", matrix[0][0])
print("False Negatives:", matrix[1][0])
print("True Positive:", matrix[1][1])
print("False Positives:", matrix[0][1])

print(classification_report(predicted, y_test))

print(np.where(predicted == 1.0))
X_test = X_test.reset_index()
#searchList = X_test['Post']

pd.set_option('display.max_colwidth', None)
searchList = searchList.drop('index', 1)
# print(searchList.loc[np.where(predicted == 1.0)])
print(searchList.loc[76])