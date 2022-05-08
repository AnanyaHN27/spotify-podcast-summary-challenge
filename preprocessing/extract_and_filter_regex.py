import os
import json
import pandas as pd
import re
import sqlite3

# instantiate a database to write the table to
connection = sqlite3.connect("podcast_test_set.db") 
crsr = connection.cursor()

# create database database 
sql_command = """CREATE TABLE dataset_ ( 
episode_id TEXT PRIMARY KEY, 
transcript TEXT,
sponsor TEXT
);"""

# get the main directory
main_dir = 'podcasts-transcripts-6to7.tar/podcasts-transcripts-6to7/spotify-podcasts-2020/podcasts-transcripts'

# does it contain urls?
def contain_url(sentence):
    x = re.search("(https\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*", sentence)
    if x == None:
	return False
    return True

# does it contain sponsors?
def contain_spon(sentence):
    spon_list = ["Anchor", "sponsor", "This podcast is sponsored by *"]
    spon_reg_list = map(re.compile, spon_list)
    return any(i.match(sentence) for i in spon_reg_list)

list_of_transcripts = []
count = 0

for filename in os.listdir(main_dir):
    for i_file in os.listdir(main_dir + "/" + filename): #6
        for j_file in os.listdir(main_dir + "/" + filename + "/" + i_file):
            for show in os.listdir(main_dir + "/" + filename + "/" + i_file + "/" + j_file):# j_file:
                if show != "show_7CoR5k5P9kalXZ7m2z0deo":
                        with open(main_dir + "/" + filename + "/" + i_file + "/" + j_file + "/" + show) as f:
                          data = json.load(f)
                        results = data['results']
                        per_ep_string = ""
                        for i in results:
                            for j in i['alternatives']:
                                if "transcript" in j:
                                    per_ep_string += j['transcript']
                                    print(j['transcript'])
                        per_ep_string.replace("\t", "")
			sponsor = (containAnchor(transcript))
			if not contain_spon(per_ep_string) and not contain_url(per_ep_string) and len(per_ep_string.split()) < 100:
          sql_command = "INSERT INTO dataset_ (episode_id, transcript) VALUES (?, ?);"
        	vals = (episode_id, transcript, sponsor)
        	crsr.execute(sql_command, vals)
                    
# write to the database and close                
connection.commit()
connection.close()
