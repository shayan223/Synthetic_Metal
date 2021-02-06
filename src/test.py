import numpy as np
import pandas as pd
import nltk
import os
from tqdm import tqdm

DATA_DIR = '../data/metal_lyrics'


tokenizer = nltk.LineTokenizer()
#read in data from every file
data = []

for subdir, dirs, files, in tqdm(os.walk(DATA_DIR)):
    #skip the formatting directory
    if(os.path.basename(subdir) == '.idea'):
        continue
    for file in files:
        filePath = subdir + os.sep + file
        #print(filePath)
        try:
            data.append(open(filePath).read())
        except:
            continue
    #NOTE DEBUGGING LINE, ONLY READ FIRST 1000 LINES
    if(len(data) > 1000):
        break
#get rid of another formatting string that gets read in
data.pop(0)

#Tokenize data line by line
#Tokens contains at each index, a list of strings, each string being
# a line of a song
tokens = []
for song in data:
    tokens.append(tokenizer.tokenize(song))
