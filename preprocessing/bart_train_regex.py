import sqlite3
import pandas as pd
import spacy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Get the data from the .db file
cnx = sqlite3.connect('podcast_test_set.db')
df = pd.read_sql_query("SELECT * FROM dataset_", cnx)
df = df[['episode_id', 'transcript']]

#Get the metadata
df_metadata = pd.read_csv('metadata.tsv', sep='\t')

#Join data properly and write to file
final_bart_data = pd.merge(df, df_metadata, left_on=  ['episode_id'],
                   right_on= ['episode_uri'], 
                   how = 'inner')[['episode_id', 'transcript', 'episode_description']]
final_bart_data.to_pickle("bart_train_regex.pkl")
