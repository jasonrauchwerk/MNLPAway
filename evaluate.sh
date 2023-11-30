# "Random" "BM25Monolingual" "BM25Translated" "Embeddings" "TranslationEmbeddings"

for retriever in "Random" "BM25Monolingual" "BM25Translated" "Embeddings" "TranslationEmbeddings"
do
    echo $retriever
    python postprocess_data.py data/SubtaskA/outputs/$retriever/predictions.jsonl data/SubtaskA/outputs/$retriever/predictions_processed.jsonl
    python ../SemEval2024-task8/subtaskA/scorer/scorer.py \
        --gold_file_path=data/SubtaskA/subtaskA_dev_multilingual.jsonl \
        --pred_file_path=data/SubtaskA/outputs/$retriever/predictions_processed.jsonl
    echo "------------------------------------"
done