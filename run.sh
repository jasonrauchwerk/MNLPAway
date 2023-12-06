# "Random" "BM25Monolingual" "BM25Translated" "Embeddings" "TranslationEmbeddings"

for retriever in "Embeddings" "TranslationEmbeddings"
do
    echo $retriever
    mkdir -p data/SubtaskA/outputs/$retriever
    python -u main.py \
        $retriever \
        "data/SubtaskA/subtaskA_dev_multilingual_processed_combined.jsonl" \
        "data/SubtaskA/outputs/$retriever/predictions_mt0.jsonl" \
        1 \
        False
done
