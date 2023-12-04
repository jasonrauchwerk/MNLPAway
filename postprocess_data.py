import json
import sys

def main(input_file: str, output_file: str):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            datum = json.loads(line)
            resp = datum['label'].lower()
            if "0" in resp:
                datum['label'] = 0
            elif "1" in resp:
                datum['label'] = 1
            elif "yes" in resp:
                datum['label'] = 1
            elif "no" in resp:
                datum['label'] = 0
            elif "human" in resp:
                datum['label'] = 0
            elif "computer" in resp:
                datum['label'] = 1
            else:
                datum['label'] = 0
            f_out.write(json.dumps(datum) + "\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
