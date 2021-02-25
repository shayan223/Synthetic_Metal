import torch
import torch.nn as nn
import tensorflow as tf

class Metal(Dataset):
    def __init__(self, control_code, gpt2_type="gpt2"):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.songs = []
            for song in Dataset:
                self.songs.append(torch.tensor(
                    self.tokenizer.encode(f"<|{control_code}|>{song}<|endoftext|>")
                ))
                
        self.songs_count = len(self.songs)
        
    def __len__(self):
        return self.songs_count

    def __getitem__(self, item):
        return self.songs[item]


def gpt_2 (data):
    torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    model = GPT2LMHeadModel.from_pretrained(data)
    outputs = model()