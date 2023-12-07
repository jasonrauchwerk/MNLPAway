# "Random" "BM25Monolingual" "BM25Translated" "Embeddings" "TranslationEmbeddings"

echo "MT0"
for retriever in "Random_0" "Random" "BM25Monolingual" "BM25Translated" "Embeddings" "TranslationEmbeddings"
do
    echo $retriever
    python subtaskA/scorer/scorer.py \
        --gold_file_path=../MNLPAway/data/SubtaskA/subtaskA_dev_multilingual_processed_combined.jsonl \
        --pred_file_path=../MNLPAway/data/SubtaskA/outputs/$retriever/predictions_mt0_processed_new.jsonl
    echo "------------------------------------"
done

echo "------------------------------------"
echo "------------------------------------"
echo "BLOOMZ"

for retriever in "Random_0" "Random" "BM25Monolingual" "BM25Translated" "Embeddings" "TranslationEmbeddings"
do
    echo $retriever
    python subtaskA/scorer/scorer.py \
        --gold_file_path=../MNLPAway/data/SubtaskA/subtaskA_dev_multilingual_processed_combined.jsonl \
        --pred_file_path=../MNLPAway/data/SubtaskA/outputs/$retriever/predictions_processed_new.jsonl
    echo "------------------------------------"
done
