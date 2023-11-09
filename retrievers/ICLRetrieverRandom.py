import random

from .ICLRetrieverBase import ICLRetrieverBase

class ICLRetrieverRandom(ICLRetrieverBase):

    def __init__(self, data):
        super().__init__(data)
        self.data             = data
        self.corpus_label_map = {datum['text']:datum['label'] for datum in data}
        self.corpus           = [datum['text'] for datum in data]
    
    def __call__(self, input_sentence: str, k: int) -> list[tuple[str, int]]:
        retrieved_documents = random.sample(self.corpus, k)
        return [(document, self.corpus_label_map[document]) for document in retrieved_documents]