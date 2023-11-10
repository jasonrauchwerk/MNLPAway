retriever="Random"

mkdir -p data/SubtaskA/outputs/$retriever
python -u main.py \
    --retriever_name $retriever \
    --test_file "data/SubtaskA/subtaskA_dev_multilingual_processed_combined.jsonl" \
    --output_file "data/SubtaskA/outputs/$retriever/predictions.jsonl" \
    --k 1 \
    --in_language False