import sqlite3
import pandas as pd
import spacy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split #split data into train and test sets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score#to handle calculating metrics for evaluation
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline #pipeline to implement steps in series
from gensim import parsing # To stem data
import nltk
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download("stopwords")
nltk.download('wordnet')
import time


start = time.time()

cnx = sqlite3.connect('podcast_test_set.db')

df = pd.read_sql_query("SELECT * FROM dataset_", cnx)

"""
Using the self-annotated test set

#df_labels = pd.read_csv('labels.csv')
#df_labels = df_labels[(df_labels.labels != "No") | (df_labels.labels != "Yes")]
#df = pd.concat([df, df_labels], axis=1)
#df = df[0: len(df_labels)]
#df = pd.read_csv('tiny_test.csv')
"""


def clean_sentence(text):
        stop_words = stopwords.words('english')
        wordnet_lemmatizer = WordNetLemmatizer()
        regex_tokenizer = RegexpTokenizer(r'\w+')
        tokenized_text = regex_tokenizer.tokenize(text)
        tokenized_text = [w.lower() for w in tokenized_text if w.isalpha()]
        tokenized_text = [w for w in tokenized_text if not w in stop_words]
        tokenized_text = [wordnet_lemmatizer.lemmatize(
            w) for w in tokenized_text]
        return tokenized_text

def tokenise(text):
    tokens = []
    sent_text = nltk.sent_tokenize(text)
    for line in sent_text:
        tokens.extend(clean_sentence(line.strip()))
    return ' '.join(str(elem) for elem in tokens)

for i in range(0,len(df)):
    df.iloc[i,1]=tokenise(df.iloc[i,1])

ep_id, X, y = df['episode_id'].tolist(), df['transcript'].tolist(), df['sponsor'].tolist() #for the auto-annotation
X_to_ep = dict(zip(X, ep_id))
#X, y = df['transcript'].tolist(), df['labels'].tolist() #using the self-annotated test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)



#Use pipeline to carry out steps in sequence with a single object
#SVM's rbf kernel gives highest accuracy in this classification problem.

logisticRegr = LogisticRegression()

print ("Training")
#train model
logisticRegr.fit(X_train, y_train)

print("Predicting")
#predict class form test data 
predicted = logisticRegr.predict(X_test)
print(predicted, y_test)
print("Macro Precision: ", precision_score(y_test, predicted, average='macro'))
print("Micro Precision: ", precision_score(y_test, predicted, average='micro'))

print("Macro Recall: ", recall_score(y_test, predicted, average='macro'))
print("Micro Recall: ", recall_score(y_test, predicted, average='micro'))

print("Accuracy: ", metrics.accuracy_score(y_test, predicted, normalize=False))
end = time.time()
print(end-start)

for prediction, i in enumerate(predicted):
    print(prediction, X_test[i], X_to_ep[X_test[i]])

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(metrics.accuracy_score(y_test, predicted, normalize=False))
plt.title(all_sample_title, size = 15);


