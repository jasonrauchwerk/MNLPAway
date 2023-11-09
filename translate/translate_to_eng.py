from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import trange
import torch
import pandas as pd
import json
import glob

model_name = 'facebook/nllb-200-distilled-600M'
# model_name = 'facebook/nllb-200-1.3B'
# model_name = 'facebook/nllb-200-3.3B'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

lang_id_map = {
    'arabic'     : 'acm_Arab',
    'russian'    : 'rus_Cryl',
    'chinese'    : 'zho_Hans',
    'indonesian' : 'ind_Latn',
    'urdu'       : 'urd_Arab',
    'bulgarian'  : 'bul_Cyrl',
    'german'     : 'deu_Latn',
}

def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    return df

def save_data(df, path):
    with open(path.replace('.jsonl', '_processed.jsonl'), 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            json_str = row.to_json()
            f.write(json_str+'\n')

def translate(data_df, src_lang):
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)

    translations = []

    data = list(data_df['text'])
    print(f'Translating from {src_lang} to English for {len(data)} datapoints')

    for i in trange(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=False, return_tensors='pt').to(device)
        generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id['eng_Latn'], max_length=400)
        translations += tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    data_df['text_english'] = translations
    return data_df

def main(data_path):
    print(f"Processing {data_path}")
    data_df = read_data(data_path)
    data_df['text_english'] = data_df['text']

    for lang, lang_id in lang_id_map.items():
        data_df[data_df['source']==lang] = translate(data_df[data_df['source']==lang], lang_id)

    save_data(data_df, data_path)
        
if __name__ == '__main__':
    main()