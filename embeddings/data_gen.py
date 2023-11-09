## run after gen_embeddings.py
import pandas as pd
import json
import glob
import plac

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

# Use generated data for each language
#   'arabic', 'russian', 'chinese', 'indonesian', 'urdu', 'bulgarian', 'german', 'english' 

# Will we have a source in the eval set?

def main(data_file: str):
    
    df = read_data(data_file)  
    save_data(df[df['source']=='arabic'], data_file, 'arabic')
    save_data(df[df['source']=='russian'], data_file, 'russian')
    save_data(df[df['source']=='chinese'], data_file, 'chinese')
    save_data(df[df['source']=='indonesian'], data_file, 'indonesian')
    save_data(df[df['source']=='urdu'], data_file, 'urdu')
    save_data(df[df['source']=='bulgarian'], data_file, 'bulgarian')
    save_data(df[df['source']=='german'], data_file, 'german')
    save_data(df[~df['source'].isin(['arabic', 'russian', 'chinese', 'indonesian', 'urdu', 'bulgarian', 'german'])], data_file, 'english')
    #-----------------------------------------------------------------------------------------------------------
    ## generate data for multilingual top k, translated top k

    save_data(df, data_file)
    #-----------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    plac.call(main) 