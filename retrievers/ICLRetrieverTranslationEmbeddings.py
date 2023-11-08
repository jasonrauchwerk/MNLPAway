from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import random
import numpy as np

from .ICLRetrieverBase import ICLRetrieverBase

embeddings_model_name = 'sentence-transformers/stsb-xlm-r-multilingual'
# translation_model_name = 'facebook/nllb-200-distilled-600M'
# translation_model_name = 'facebook/nllb-200-1.3B'
translation_model_name = 'facebook/nllb-200-3.3B'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lang_id_map = {
    'arabic'     : 'acm_Arab',
    'russian'    : 'rus_Cryl',
    'chinese'    : 'zho_Hans',
    'indonesian' : 'ind_Latn',
    'urdu'       : 'urd_Arab',
    'bulgarian'  : 'bul_Cyrl',
    'german'     : 'deu_Latn',
}


class ICLRetriever(ICLRetrieverBase):
    # topk monolingual embeddings
    # topk multilingual embeddings
    # topk translated embeddings
    
    def __init__(self, data):
        super().__init__(data)
        self.data             = data
        self.corpus_label_map = {datum['text']:(datum['label']) for datum in data}
        self.corpus_embedding_map = {datum['embedding']:(datum['text']) for datum in data}
        self.data_embedding         = [datum['embedding'] for datum in data]
    
    def _gen_embeddings(self, sentence, input_language):
        model = SentenceTransformer(embeddings_model_name).to(device) 
        embeddings = model.encode(sentence)
        return embeddings
    
    def _translate(self, input_sentence, input_language):
        model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
                translation_model_name, src_lang=lang_id_map[input_language]
            )
        inputs = tokenizer(input_sentence, return_tensors='pt').to(device)
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id['eng_Latn'], max_length=400)
        translated_sentence = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_sentence
    
    def __call__(self, input_sentence: str, k: int, input_language: str):# -> list[tuple[str, int]]:
        '''
        Args
            input_sentence (str): the sentence for inference
            k (int): number of examples to be selected for ICL
            input_language: the language of input_sentence. 
                Permitted values - ['arabic','russian', 'chinese', 'indonesian','urdu', 
                                    'bulgarian','german','english']
                            
        '''
        input_embeddings = self._gen_embeddings([self._translate(input_sentence, input_language)])
        input_embeddings = np.array(input_embeddings)
        self.data_embedding = np.array(self.data_embedding)
        
        cosine_similarities = cosine_similarity(input_embeddings.reshape(1, -1), self.data_embedding)
        
        cs_embedding_list = []
        for (cs,embedding) in zip(cosine_similarities[0], self.data_embedding):
            cs_embedding_list.append(cs, embedding)
        
        cs_embedding_list = sorted(cs_embedding_list, key = lambda x:x[0])
        
        result = []
        for i in cs_embedding_list[:k]:
            text = self.corpus_embedding_map(i[1])
            result.append(
                (text, self.corpus_label_map(text))
            )
        
        return result