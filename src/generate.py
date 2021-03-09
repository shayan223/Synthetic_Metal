from transformers import GPT2Tokenizer
import filter
import torch
from tqdm import tqdm
import os
from os import path
import json
from torch.utils.data import Dataset, DataLoader

# gpt2_small_chunks = 1280
# gpt2_medium_chunks = 1280
# gpt2_large_chunks = 1280
gpt2_xl_chunks = 1600


class lyrics_generator(Dataset):

    def __init__(self, data, control_code="song", gpt2_type="gpt2", max_length=gpt2_xl_chunks, load_from_file=True, save_to_file=True):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.songs = []

        if load_from_file:
            if not self.load_encoded_data():
                # Append byte-tokenized songs to list in tensor form
                for song in tqdm(data):
                    if len(song) < gpt2_xl_chunks:
                        self.songs.append(torch.tensor(
                            self.tokenizer.encode("<|{}|>{}<|endoftext|>".format(control_code, song))
                        ))

                # Gets the number of songs loaded in
                self.song_count = len(self.songs)

                # Saves the byte-encoded GPT2-tokenized tensors to a file
                if save_to_file:
                    f_data = {'songs': self.songs}
                    with open('../data/song_tensors.dat', 'w') as f_out:
                        json.dump(f_data, f_out)


    def load_encoded_data(self):
        fpath = "../data/song_tensors.dat"

        # Load the filtered song list if it exists, read over corpus and create it.
        if path.exists(fpath):
            with open(fpath) as f:
                self.songs = json.load(f)['songs']
                return True
        return False

    def __len__(self):
        return self.song_count

    def __getitem__(self, item):
        return self.songs[item]