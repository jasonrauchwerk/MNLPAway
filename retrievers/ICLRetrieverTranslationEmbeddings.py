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

class ICLRetrieverTranslationEmbeddings(ICLRetrieverBase):
    # topk translated embeddings
    
    def __init__(self, data):
        super().__init__(data)
        self.data             = data
        self.corpus_label_map = {datum['text']:(datum['label']) for datum in data}
        self.corpus_embedding_map = {tuple(datum['text_english_embeddings']):datum['text'] for datum in data}
        self.data_embedding         = np.array([datum['text_english_embeddings'] for datum in data])
        self.lang_id_map = {
            'arabic'     : 'acm_Arab',
            'russian'    : 'rus_Cryl',
            'chinese'    : 'zho_Hans',
            'indonesian' : 'ind_Latn',
            'urdu'       : 'urd_Arab',
            'bulgarian'  : 'bul_Cyrl',
            'german'     : 'deu_Latn',
            'english'    : 'eng_Latn',
        }
    
    def _gen_embeddings(self, sentence):
        model = SentenceTransformer(embeddings_model_name).to(device) 
        embeddings = model.encode(sentence)
        return embeddings
    
    def _translate(self, input_sentence, input_language):
        model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
                translation_model_name, src_lang=self.lang_id_map[input_language]
            )
        inputs = tokenizer(input_sentence, return_tensors='pt').to(device)
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id['eng_Latn'], max_length=400)
        translated_sentence = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_sentence
    
    def __call__(self, datum, input_sentence: str, k: int):# -> list[tuple[str, int]]:
        '''
        Args
            input_sentence (str): the sentence for inference
            k (int): number of examples to be selected for ICL
            input_language: the language of input_sentence. 
                Permitted values - ['arabic','russian', 'chinese', 'indonesian','urdu', 
                                    'bulgarian','german','english']
                            
        '''
        # input_language = datum['source'] if datum['source'] in self.lang_id_map else 'english'
        # input_embeddings = self._gen_embeddings([self._translate(input_sentence, input_language)])
        # input_embeddings = np.array(input_embeddings)
        input_embeddings = np.array(datum['text_english_embeddings'])        
        cosine_similarities = cosine_similarity(input_embeddings.reshape(1, -1), self.data_embedding)
        
        if k == 1:
            max_cs = -10
            max_cs_embedding = None
            for (cs,embedding) in zip(cosine_similarities[0], self.data_embedding):
                if cs > max_cs:
                    max_cs_embedding = (cs, embedding)
                    max_cs = cs
            text = self.corpus_embedding_map[tuple(max_cs_embedding[1])]
            result = []
            result.append((text, self.corpus_label_map[text]))
        
        else:        
            cs_embedding_list = []
            for (cs,embedding) in zip(cosine_similarities[0], self.data_embedding):
                cs_embedding_list.append((cs, embedding))
            
            cs_embedding_list = sorted(cs_embedding_list, key = lambda x:-x[0])
            
            result = []
            for i in cs_embedding_list[:k]:
                text = self.corpus_embedding_map(i[1])
                result.append(
                    (text, self.corpus_label_map(text))
                )
        
        return datum, input_sentence, result