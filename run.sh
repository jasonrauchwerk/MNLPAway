# "Random" "BM25Monolingual" "BM25Translated" "Embeddings" "TranslationEmbeddings"
# "bigscience/mt0-large" "bigscience/bloomz-1b7" "bigscience/bloomz-3b"

export checkpoint="bigscience/mt0-large"
echo $checkpoint
for retriever in "Random"
do
    echo $retriever
    mkdir -p data/SubtaskA/outputs/$retriever
    python -u main.py \
        $retriever \
        "data/SubtaskA/subtaskA_dev_multilingual_processed_combined.jsonl" \
        "data/SubtaskA/outputs/$retriever/predictions_mt0.jsonl" \
        1 \
        False \
        $checkpoint
done

export checkpoint="bigscience/mt0-large"
echo $checkpoint
for retriever in "Random"
do
    echo $retriever
    mkdir -p data/SubtaskA/outputs/$retriever"_0"
    python -u main.py \
        $retriever \
        "data/SubtaskA/subtaskA_dev_multilingual_processed_combined.jsonl" \
        "data/SubtaskA/outputs/"$retriever"_0/predictions_mt0.jsonl" \
        0 \
        False \
        $checkpoint
done

export checkpoint="bigscience/bloomz-1b7"
echo $checkpoint
for retriever in "Random"
do
    echo $retriever
    mkdir -p data/SubtaskA/outputs/$retriever"_0"
    python -u main.py \
        $retriever \
        "data/SubtaskA/subtaskA_dev_multilingual_processed_combined.jsonl" \
        "data/SubtaskA/outputs/"$retriever"_0/predictions.jsonl" \
        0 \
        False \
        $checkpoint
done