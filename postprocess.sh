# "Random" "BM25Monolingual" "BM25Translated" "Embeddings" "TranslationEmbeddings"

for retriever in "Random" "BM25Monolingual" "BM25Translated" "Embeddings" "TranslationEmbeddings"
do
    echo $retriever
    python postprocess_data.py \
        data/SubtaskA/outputs/$retriever/predictions.jsonl \
        data/SubtaskA/outputs/$retriever/predictions_processed_new.jsonl
    echo "------------------------------------"
done
