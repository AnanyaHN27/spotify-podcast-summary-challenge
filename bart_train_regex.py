import sqlite3
import pandas as pd
import spacy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

cnx = sqlite3.connect('podcast_test_set.db')
df = pd.read_sql_query("SELECT * FROM dataset_", cnx)
df = df[['episode_id', 'transcript']]

df_metadata = pd.read_csv('metadata.tsv', sep='\t')
#final_bart_data = pd.merge(df, df_metadata, on='episode_uri', how='inner')

final_bart_data = pd.merge(df, df_metadata, left_on=  ['episode_id'],
                   right_on= ['episode_uri'], 
                   how = 'inner')[['episode_id', 'transcript', 'episode_description']]

print(final_bart_data.iloc[0]['episode_id'])
print(final_bart_data.iloc[0]['episode_description'])


final_bart_data.to_pickle("bart_train_regex.pkl")

