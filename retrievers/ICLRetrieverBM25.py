from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import random

from .ICLRetrieverBase import ICLRetrieverBase

class ICLRetrieverBM25(ICLRetrieverBase):
    # random monolingual
    # random multilingual
    # bm25 monolingual
    # bm25 translated

    def __init__(self, data):
        super().__init__(data)
        self.data             = data
        self.corpus_label_map = {datum['text']:datum['label'] for datum in data}
        self.corpus           = [datum['text'] for datum in data]
        self.bm25             = BM25Okapi(self.corpus, self._tokenizer)
        
    def _tokenizer(self, text):
        return word_tokenize(text.lower())
    
    def __call__(self, input_sentence: str, k: int):# -> list[tuple[str, int]]:
        # Randomly select one document 
        if k==-1: retrieved_documents = [random.choice(self.corpus)]
        
        # Select top-k documents using BM25
        else: retrieved_documents = self.bm25.get_top_n(self._tokenizer(input_sentence), self.corpus, k)
        
        return [(document, self.corpus_label_map[document]) for document in retrieved_documents]