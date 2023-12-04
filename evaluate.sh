# "Random" "BM25Monolingual" "BM25Translated" "Embeddings" "TranslationEmbeddings"

for retriever in "Random" "BM25Monolingual" "BM25Translated" "Embeddings" "TranslationEmbeddings"
do
    echo $retriever
    python subtaskA/scorer/scorer.py \
        --gold_file_path=../MNLPAway/data/SubtaskA/subtaskA_dev_multilingual_processed_combined.jsonl \
        --pred_file_path=../MNLPAway/data/SubtaskA/outputs/$retriever/predictions_processed.jsonl
    echo "------------------------------------"
done
