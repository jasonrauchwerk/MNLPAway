import json
import re

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import plac
from tqdm import tqdm
import torch
import gc

from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
MAX_THREADS = 64

from retrievers.ICLRetrieverBM25 import ICLRetrieverBM25Monolingual, ICLRetrieverBM25Translated
from retrievers.ICLRetrieverRandom import ICLRetrieverRandom
from retrievers.ICLRetrieverEmbeddings import ICLRetrieverEmbeddings
from retrievers.ICLRetrieverTranslationEmbeddings import ICLRetrieverTranslationEmbeddings


def construct_prompt(text, exemplars, verbose=False):
    prompt = "Respond 0 or 1 to determine whether the following paragraph was generated by a human or computer. Respond 0 if it is human-generated or 1 if it is computer-generated. Only respond with either the number 0 or 1."
    truncate_length = int(3e4 // len(exemplars))
    if exemplars:
        prompt += "\nFor Example,\n"
    for exemplar, label in exemplars:
        if len(exemplar) > truncate_length:
            print(f"\nTruncating exemplar of {len(exemplar)} length to {truncate_length} \n")
            exemplar = exemplar[:truncate_length]
        if verbose: print(f"\nLengths exemplar - {len(exemplar)}")
        prompt += f"Paragraph: {exemplar}\nResponse: {label}\n"
    if verbose: print(f"Lengths text - {len(text)}")
    prompt += f"Paragraph: {text}\nResponse: "
    if verbose: print(f"Lengths prompt - {len(prompt)}")
    if verbose: print('-'*20)
    return prompt

def extract_label(text):
    response = re.findall("Response: (.*)$", text, flags=re.MULTILINE)
    if response:
        return response[-1]
    else:
        return -1

def main(retriever_name: str, test_file: str, output_file: str, k: int, in_language: bool):
    in_language = False
    checkpoint = "bigscience/mt0-large"
    # checkpoint = "bigscience/bloomz-1b7"
    # checkpoint = "bigscience/bloomz-3b"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
    print("Model and Tokenizer Loaded")

    retriever_dict = {"BM25Monolingual"       : ICLRetrieverBM25Monolingual,
                      "BM25Translated"        : ICLRetrieverBM25Translated,
                      "Random"                : ICLRetrieverRandom,
                      "Embeddings"            : ICLRetrieverEmbeddings,
                      "TranslationEmbeddings" : ICLRetrieverTranslationEmbeddings}
    
    if in_language:
        langs = ['arabic', 'russian', 'chinese', 'indonesian', 'urdu', 'bulgarian', 'german', 'english']
        retrievers = {}
        for language in langs:
            train_file = f"data/SubtaskA/subtaskA_train_multilingual_{language}.jsonl"
            with open(train_file, 'r') as f:
                train_data = [json.loads(line) for line in f]
            retrievers[language] = retriever_dict[retriever_name](train_data)
    else:  # cross-lingual
        train_file = "data/SubtaskA/subtaskA_train_multilingual_processed_combined.jsonl"
        with open(train_file, 'r') as f:
            train_data = [json.loads(line) for line in f]
        retriever = retriever_dict[retriever_name](train_data)
        print("Retriever Loaded")

    cnt = 0
    with open(test_file, 'r') as f_in:
        for line in f_in: 
            cnt += 1

    with open(test_file, 'r') as f_in, open(output_file, 'w') as f_out:
        texts = []
        for line in tqdm(f_in, total=cnt, ncols=100):
            datum = json.loads(line)
            text = datum['text']
            texts.append((datum, text))

        exemplars_list = []
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = []
            for (datum, text) in tqdm(texts, desc='Submitting Jobs', ncols=100):
                if in_language:
                    if datum['source'] in retrievers:
                        retriever = retrievers[datum['source']]
                    else:
                        retriever = retrievers['english']
                futures.append(executor.submit(retriever, datum, text, int(k)))

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Getting Retriever Results", ncols=100):
                exemplar = future.result()
                exemplars_list.append(exemplar)
            
        for (datum, text, exemplars) in tqdm(exemplars_list, ncols=100, desc="Inference"):
            with torch.inference_mode():
                inputs = tokenizer.encode(construct_prompt(text, exemplars), return_tensors="pt").to("cuda")
                outputs = model.generate(inputs, max_length = 4000)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            label = extract_label(response)
            f_out.write(json.dumps({"id": datum['id'], "label": label}) + "\n")

if __name__ == "__main__":
    plac.call(main)
