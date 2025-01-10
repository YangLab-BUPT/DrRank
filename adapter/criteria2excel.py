import json

with open("adapter/DrRank_criteria.Qwen2.5-7B-Instruct.json") as f:
    data = json.load(f)

alist = []
for k in data.keys():
    adict = dict()
    adict["疾病-治疗方式"] = k
    adict["评估标准"] = data[k]
    alist.append(adict)

import pandas as pd

df = pd.DataFrame(alist)
df.to_excel("adapter/DrRank_criteria.Qwen2.5-7B-Instruct.xlsx", index=False)
