from translate import translate_to_eng
from embeddings import gen_embeddings, data_gen

if __name__ == '__main__':
    # translate_to_eng.main('data/SubtaskA/subtaskA_train_multilingual.jsonl')
    gen_embeddings.main('data/SubtaskA/subtaskA_train_multilingual_processed.jsonl')
    data_gen.main('data/SubtaskA/subtaskA_train_multilingual_processed.jsonl')