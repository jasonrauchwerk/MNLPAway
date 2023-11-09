from translate import translate_to_eng
from embeddings import gen_embeddings, data_gen

if __name__ == '__main__':
    split = 'dev'
    translate_to_eng.main(f'data/SubtaskA/subtaskA_{split}_multilingual.jsonl')
    gen_embeddings.main(f'data/SubtaskA/subtaskA_{split}_multilingual_processed.jsonl')
    data_gen.main(f'data/SubtaskA/subtaskA_{split}_multilingual_processed.jsonl')