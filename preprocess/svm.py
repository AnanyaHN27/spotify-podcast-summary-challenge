import sqlite3
import pandas as pd
import spacy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split #split data into train and test sets
from sklearn.feature_extraction.text import CountVectorizer #convert text comment into a numeric vector
from sklearn.feature_extraction.text import TfidfTransformer #use TF IDF transformer to change text vector created by count vectorizer
from sklearn.svm import SVC# Support Vector Machine
from sklearn.metrics import precision_score#to handle calculating metrics for evaluation
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline #pipeline to implement steps in series
from gensim import parsing # To stem data
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download("stopwords")
nltk.download('wordnet')
import time

start = time.time()

def hybrid_using_auto_annotated():
        cnx = sqlite3.connect('podcast_test_set.db')
        df = pd.read_sql_query("SELECT * FROM dataset_", cnx)
        return df

#Using the self-annotated test set

def using_self_annotated():
        df_labels = pd.read_csv('labels.csv')
        df_labels = df_labels[(df_labels.labels != "No") | (df_labels.labels != "Yes")]
        df = pd.concat([df, df_labels], axis=1)
        df = df[0: len(df_labels)]
        df = pd.read_csv('tiny_test.csv')
        return df


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

df = using_self_annotated()

ep_id, X, y = df['episode_id'].tolist(), df['transcript'].tolist(), df['sponsor'].tolist() #for the auto-annotation
X_to_ep = dict(zip(X, ep_id))

for i in range(0,len(df)):
    df.iloc[i,1]=tokenise(df.iloc[i,1])
    
#X, y = df['transcript'].tolist(), df['labels'].tolist() #using the self-annotated test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=11)
X_dev, X_train = train_test_split(X_train, test_size=0.78, random_state=11)

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(kernel='rbf'))])

print ("Training")
#train model
text_clf.fit(X_train, y_train)

print("Predicting")
#predict class from test data 
predicted = text_clf.predict(X_dev)
print(predicted, y_test)
print("Macro Precision: ", precision_score(y_test, predicted, average='macro'))
print("Micro Precision: ", precision_score(y_test, predicted, average='micro'))

print("Macro Recall: ", recall_score(y_test, predicted, average='macro'))
print("Micro Recall: ", recall_score(y_test, predicted, average='micro'))
end = time.time()
print(end-start)

def get_sponsored_episodes(predicted, X_test, X_to_ep):
        filter_out = []
        for i, prediction in enumerate(predicted):
            if prediction == "Yes":
                filter_out.append(X_to_ep[X_test[i]])
        return filter_out

filter_out = get_sponsored_episodes(predicted, X_test, X_to_ep)
                
episode_id_list = []
transcript_list = []

for i, prediction in enumerate(predicted):
        if prediction == "No":
                episode_id_list.append(X_to_ep[X_test[i]])
                transcript_list.append(X_test[i])

intermediate_dict = {'episode_id': episode_id_list, 'transcript': transcript_list}
df_intermediate = pd.DataFrame(intermediate_dict)

df_metadata = pd.read_csv('metadata.tsv', sep='\t')
final_bart_data = pd.merge(df_intermediate, df_metadata, on='episode_id', how='inner')

final_bart_data.to_pickle("bart_train_svm.pkl")

df_filter = pd.DataFrame(filter_out)
df.to_csv('eps_to_filter.csv', index=False)
