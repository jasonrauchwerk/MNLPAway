class ICLRetrieverBase:
    # random monolingual
    # random multilingual
    # bm25 monolingual
    # bm25 translated
    # topk monolingual embeddings
    # topk multilingual embeddings
    # topk translated embeddings

    # translate with nllb, mt5
    # multilingual embeddings: sentence-transformers/stsb-xlm-r-multilingual
    def __init__(self, data):
        #{"text":str ,"label":int ,"model": str,"source": str,"id": int}
        # possible values of source: {'arxiv', 'wikihow', 'reddit', 'chinese', 'peerread', 'indonesian', 'bulgarian', 'wikipedia', 'urdu'}
        pass
    
    def __call__(self, input_sentence: str, k: int):# -> list[tuple[str, int]]:
        # returns [(text1, label1), (text2, label2), ..., (textk, labelk)]
        raise NotImplementedError