## run after gen_embeddings.py
import pandas as pd
import json
import glob

def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    return df

def save_data(df, path, language="combined"):
    with open(path.replace('.jsonl', f'_{language}.jsonl'), 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            json_str = row.to_json()
            f.write(json_str+'\n')

#-----------------------------------------------------------------------------------------------------------
## generate data for monolingual top k

#if language is english, use dev_monolingual

# for other languages, use generated data
#   'arabic', 'russian', 'chinese', 'indonesian', 'urdu', 'bulgarian', 'german'    
for data_path in glob.glob('data/*multilingual.jsonl'):
    df = read_data(data_path)
    save_data(df[df['source']=='arabic'], data_path, 'arabic')
    save_data(df[df['source']=='russian'], data_path, 'russian')
    save_data(df[df['source']=='chinese'], data_path, 'chinese')
    save_data(df[df['source']=='indonesian'], data_path, 'indonesian')
    save_data(df[df['source']=='urdu'], data_path, 'urdu')
    save_data(df[df['source']=='bulgarian'], data_path, 'bulgarian')
    save_data(df[df['source']=='german'], data_path, 'german')
#-----------------------------------------------------------------------------------------------------------
## generate data for multilingual top k, translated top k

df1 = read_data(glob.glob('data/*multilingual.jsonl'))
df2 = read_data(glob.glob('data/*monolingual.jsonl'))

save_data(pd.concat([df1, df2]),glob.glob('data/*multilingual.jsonl'))
#-----------------------------------------------------------------------------------------------------------
