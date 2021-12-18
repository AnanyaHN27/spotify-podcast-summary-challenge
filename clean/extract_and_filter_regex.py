import json
from pathlib import Path
import sys
import re
import sqlite3 

connection = sqlite3.connect("podcast_test_set.db") #instantiate a database to write the table to
crsr = connection.cursor()

# SQL command to create a table in the database 
sql_command = """CREATE TABLE dataset_ ( 
episode_id TEXT PRIMARY KEY, 
transcript TEXT,
sponsor TEXT
);"""

# execute the statement 
crsr.execute(sql_command)

def contain_url(sentence):
    x = re.search("((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*", sentence)
    return not x==None

def contain_spon(sentence):
    spon_list = ["Anchor: The easiest way to make a podcast.", "anchor", "sponsor", "This podcast is sponsored by *"]
    spon_reg_list = map(re.compile, spon_list)
    return any(i.match(sentence) for i in spon_reg_list)

def add_to_sql(podcast):
    """
    This function takes in the podcast's json transcript file and concatentates all the transcript
    segments to form a long running transcript. Information regarding when each word is uttered is 
    discarded. It builds an sql table that is then written to a .db file.
    """
    transcript = ""
    with open(str(podcast)) as json_file:
        data = json.load(json_file)
        for item in data['results']:
            try:
                transcript += ' ' + item['alternatives'][0]['transcript']
            except:
                pass #not all item with 'alternatives' key contain a 'transcript' key

    sponsor = (containAnchor(transcript))
            
    episode_id = "spotify:episode:" + Path(podcast).stem #This is the format of the episode_uri in the metadata.csv file
    if not contain_spon(transcript) and not contain_url(transcript) and len(transcript.split()) < 100:
        sql_command = "INSERT INTO dataset_ (episode_id, transcript) VALUES (?, ?);"
        vals = (episode_id, transcript, sponsor)
        crsr.execute(sql_command, vals)


def podcasts_to_df(directory_path):
    """
    This function takes in the folder where the dataset is located and builds a table for each
    podcast transcript file in the folder.
    """
    rootdir = Path(directory_path)
    file_list = [f for f in rootdir.glob('**/*') if f.is_file()]
    
    for file in file_list:
        add_to_sql(file)

podcasts_to_df('podcasts-transcripts-3to5.tar/podcasts-transcripts-3to5/spotify-podcasts-2020/podcasts-transcripts')
connection.commit()
connection.close()

