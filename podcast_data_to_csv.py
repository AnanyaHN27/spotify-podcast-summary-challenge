import os
import json
import pandas as pd

main_dir = 'podcasts-transcripts-3to5.tar/podcasts-transcripts-3to5/spotify-podcasts-2020/podcasts-transcripts'

list_of_transcripts = []
count = 0

for filename in os.listdir(main_dir):
    for i_file in os.listdir(main_dir + "/" + filename): #6
        #print(i_file)
        #if i_file == "6":
        for j_file in os.listdir(main_dir + "/" + filename + "/" + i_file):
            #print(j_file)
            for show in os.listdir(main_dir + "/" + filename + "/" + i_file + "/" + j_file):# j_file:
                if show != "show_7CoR5k5P9kalXZ7m2z0deo":
                    #print("show", show)
                    #count+=1
                    print(i_file, j_file, show)
                    with open(main_dir + "/" + filename + "/" + i_file + "/" + j_file + "/" + show) as f:
                      data = json.load(f)
                    results = data['results']
                    per_ep_string = ""
                    for i in results:
                        for j in i['alternatives']:
                            if "transcript" in j:
                                per_ep_string += j['transcript']
                                
                    per_ep_string.replace("\t", "")
                    print(per_ep_string)
                    list_of_transcripts.append(per_ep_string)
                    
                

df = pd.DataFrame(list_of_transcripts)
df.to_csv('raw_transcripts_all.csv', index = False, header = True, sep = '\t')
    
print(count)
