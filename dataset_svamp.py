import json
import pandas as pd
from datasets import concatenate_datasets, load_dataset, load_from_disk
from datasets import Dataset

# preprocess svamp s.t. it has similar format as multiarith
data = json.load(open('data/raw_SVAMP.json'))
# change column names, map 'Body' to 'question', etc.
new_d = {'Body': 'question', 'Answer': 'answer', 'Equation': 'equation', 'Type': 'type'}
for row in data:
    for k,v in new_d.items():
        for name in list(row):
            if name == 'Body' and k==name:
                # map 'Question' and 'Body' to 'questions'
                row[v] = row.pop(name)+row.pop('Question')
            elif name == 'Answer' and k == name:    
                row[v] = [row.pop(name)]
            elif k == name:
                row[v] = row.pop(name)
# with open("data/svamp.json", "w") as f:
#     f.write(json.dumps(data))
data = pd.DataFrame(data)
dataset = Dataset.from_pandas(data)
dataset.save_to_disk("data/svamp")
data = load_from_disk("data/svamp")
# data = (open('data/svamp.json'))
print("="*10)
print(data[0])
# dataset = Dataset.load_from_disk('data/multiarith')
# for data in dataset:
#     print(data)
#     break