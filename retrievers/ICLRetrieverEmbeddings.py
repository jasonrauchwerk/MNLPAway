from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import random
import numpy as np
import time

from .ICLRetrieverBase import ICLRetrieverBase

model_name = 'sentence-transformers/stsb-xlm-r-multilingual'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ICLRetrieverEmbeddings(ICLRetrieverBase):
    # topk monolingual embeddings
    # topk multilingual embeddings

    
    def __init__(self, data):
        super().__init__(data)
        self.data             = data
        self.corpus_label_map = {datum['text']:(datum['label']) for datum in data}
        self.corpus_embedding_map = {tuple(datum['text_embeddings']): datum['text'] for datum in data}
        self.data_embedding         = np.array([datum['text_embeddings'] for datum in data])
        

    def gen_embeddings(self, sentence):
        model = SentenceTransformer(model_name).to(device) 
        embeddings = model.encode(sentence)
        return embeddings
    
    def __call__(self, datum, input_sentence: str, k: int):# -> list[tuple[str, int]]:
        
        # input_embeddings = self.gen_embeddings([input_sentence])
        # input_embeddings = np.array(input_embeddings)
        input_embeddings = np.array(datum['text_embeddings'])
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
                text = self.corpus_embedding_map[tuple(i[1])]
                result.append(
                    (text, self.corpus_label_map[text])
                )
        
        return datum, input_sentence, result