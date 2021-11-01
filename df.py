import sqlite3
import pandas as pd
import spacy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split #split data into train and test sets
from sklearn.feature_extraction.text import CountVectorizer #convert text comment into a numeric vector
from sklearn.feature_extraction.text import TfidfTransformer #use TF IDF transformer to change text vector created by count vectorizer
from sklearn.svm import SVC# Support Vector Machine
from sklearn.pipeline import Pipeline #pipeline to implement steps in series
from gensim import parsing # To stem data
import time

start = time.time()
cnx = sqlite3.connect('podcast_test_set.db')

df = pd.read_sql_query("SELECT * FROM dataset_", cnx)

#do more here!
def parse(s):
    parsing.stem_text(s)
    return s

for i in range(0,len(df)):
    df.iloc[i,2]=parse(df.iloc[i,2])

X, y = df['transcript'].tolist(), df['sponsor'].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

labels = np.unique(y)
print(labels)

#Use pipeline to carry out steps in sequence with a single object
#SVM's rbf kernel gives highest accuracy in this classification problem.
print("1")

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(kernel='rbf'))])

print ("Training")
#train model
text_clf.fit(X_train, y_train)

print("Predicting")
#predict class form test data 
predicted = text_clf.predict(X_test)
end = time.time()
print(start-end)

#df.to_csv(r'podcast_test_set.csv', index = False, header = True)

#df = pd.read_csv("podcast_testset.csv", header=None)

#print(df.head(300))


