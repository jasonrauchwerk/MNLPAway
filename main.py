# pip install -q transformers accelerate
from retrievers.ICLRetrieverBM25 import ICLRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer
import plac
import json
import re

def construct_prompt(text, exemplars):
    prompt = "Respond 0 or 1 to determine whether the following paragraph was generated by a human or computer. Respond 0 if it is human-generated or 1 if it is computer-generated. Only respond with either the number 0 or 1."
    if exemplars:
        prompt += "\nFor Example,\n"
    for exemplar, label in exemplars:
        prompt += f"Paragraph: {exemplar}\nResponse: {label}\n"
    prompt += f"Paragraph: {text}\nResponse: "
    return prompt

def extract_label(text):
    response = re.findall("Response: (.*)$", text, flags=re.MULTILINE)
    if response:
        return response[-1]
    else:
        return -1

def main(retriever_name: str, train_file: str, test_file: str, output_file: str, k: int):
    checkpoint = "bigscience/bloomz-3b"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

    with open(train_file, 'r') as f:
        train_data = [json.loads(line) for line in f]

    retriever_dict = {"BM25": ICLRetriever}
    retriever = retriever_dict[retriever_name](train_data)

    with open(test_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            datum = json.loads(line)
            text = datum['text']
            exemplars = retriever(text, int(k))

            inputs = tokenizer.encode(construct_prompt(text, exemplars), return_tensors="pt").to("cuda")
            outputs = model.generate(inputs, max_length = 4000)
            response = tokenizer.decode(outputs[0])

            label = extract_label(response)
            f_out.write(json.dumps({"id": datum['id'], "label": label}) + "\n")

if __name__ == "__main__":
    plac.call(main)