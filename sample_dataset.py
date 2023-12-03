import sys
from collections import defaultdict
import json
import random

data = defaultdict(list)
with open(sys.argv[1], 'r') as f:
    for line in f:
        j = json.loads(line)
        data[(j['source'], j['label'])].append(line)

for _, li in data.items():
    random.shuffle(li)

with open(sys.argv[2], 'w') as f:
    for _, li in data.items():
        for line in li[::4]:
            f.write(line)
