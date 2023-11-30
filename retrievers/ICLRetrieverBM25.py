from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
import random

from .ICLRetrieverBase import ICLRetrieverBase

class ICLRetrieverBM25Monolingual(ICLRetrieverBase):

    def __init__(self, data):
        super().__init__(data)
        self.data             = data
        self.corpus_label_map = {datum['text']:datum['label'] for datum in data}
        self.corpus           = [datum['text'] for datum in data]
        self.tokenizer        = AutoTokenizer.from_pretrained("bigscience/bloomz-3b")
        self.tokenize_func    = self.tokenizer.tokenize
        self.bm25             = BM25Okapi(self.corpus, self.tokenize_func)
    
    def __call__(self, datum, input_sentence: str, k: int) -> list[tuple[str, int]]:
        retrieved_documents = self.bm25.get_top_n(self.tokenize_func(input_sentence), self.corpus, k)
        return datum, input_sentence, [(document, self.corpus_label_map[document]) for document in retrieved_documents]

class ICLRetrieverBM25Translated(ICLRetrieverBase):

    def __init__(self, data):
        super().__init__(data)
        self.data             = data
        self.corpus_label_map = {datum['text_english']:datum['label'] for datum in data}
        self.corpus           = [datum['text_english'] for datum in data]
        self.tokenizer        = AutoTokenizer.from_pretrained("bigscience/bloomz-3b")
        self.tokenize_func    = self.tokenizer.tokenize
        self.bm25             = BM25Okapi(self.corpus, self.tokenize_func)
    
    def __call__(self, datum, input_sentence: str, k: int) -> list[tuple[str, int]]:
        retrieved_documents = self.bm25.get_top_n(self.tokenize_func(input_sentence), self.corpus, k)
        return datum, input_sentence, [(document, self.corpus_label_map[document]) for document in retrieved_documents]