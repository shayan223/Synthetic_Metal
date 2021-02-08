import numpy as np
import pandas as pd
import nltk
import os
from tqdm import tqdm
from nltk.tokenize import word_tokenize


DATA_DIR = '../data/metal_lyrics'
nltk.download('words')
nltk.download('punkt')


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
filter_tokens = ['ich', 'en', 'se', 'z', 'Z', 'bir', 'ruh', 'ein', 'es', 'zur', 'te', 'y', 'di']
filter_whole = ['Ã']
for song in data:
    add_song = True
    tks = word_tokenize(song)
    if '[Instrumental]' in song or '[instrumental]' in song:
        add_song = False
    for f in filter_whole:
        if f in song:
            add_song = False
    if add_song:
        for f in filter_tokens:
            if f in tks:
                add_song = False
        if add_song:
            if song != "\n":
                tokens.append(song)
                print(song)
