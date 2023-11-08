import pandas as pd
from tqdm import trange
from sentence_transformers import SentenceTransformer
import torch
import json
import glob


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

def gen_embeddings(df, read_column):
    model = SentenceTransformer(model_name).to(device) 
    data = list(df[read_column])
    embeddings = []
    for i in data:
        embedding = model.encode(i)
        embeddings.apppend(embedding)
        df[embeddings] = embeddings

                
def main():    
    for data_path in glob.glob('data/*.jsonl'):
        data_df = read_data(data_path)
        if 'translated' in data_path:
            gen_embeddings(data_df,'text_english')
        else:
            gen_embeddings(data_df,'text')


if __name__ == '__main__':
    main() 
    
    