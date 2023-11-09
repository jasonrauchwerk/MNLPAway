#run after translate_to_eng.py

import pandas as pd
from tqdm import trange, tqdm
from sentence_transformers import SentenceTransformer
import torch
import json
import glob
import plac


model_name = 'sentence-transformers/stsb-xlm-r-multilingual'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    return df

def save_data(df, path):
    with open(path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            json_str = row.to_json()
            f.write(json_str+'\n')

def gen_embeddings(df, read_column, write_column):
    model = SentenceTransformer(model_name).to(device) 
    data = list(df[read_column])
    embeddings = []
    for i in tqdm(data, ncols=100, leave=False):
        embedding = model.encode(i)
        embeddings.append(embedding)
    df[write_column] = embeddings

                
def main(data_file: str):    
    data_df = read_data(data_file)
    gen_embeddings(data_df,'text_english', 'text_english_embeddings')
    gen_embeddings(data_df,'text', 'text_embeddings')

    save_data(data_df, data_file)


if __name__ == '__main__':
    plac.call(main) 
    
    