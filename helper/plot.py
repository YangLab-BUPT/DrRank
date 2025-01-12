import json

with open("runs/DrRank_V2.Qwen2.5-7B-Instruct.criteria.L5.jsonl") as f:
    ground_truths = []
    predict_scores = []
    for line in f:
        item = json.loads(line)
        ground_truths.extend(item["pos_scores"])
        predict_scores.extend(item["scores"])
    assert len(ground_truths) == len(predict_scores)

print(f"{len(ground_truths)} query-doctor pairs loaded")

print(f"{set(ground_truths)} unique ground_truth labels found")

predict_labels = predict_scores[0].keys()
print(f"{predict_labels=}")


import numpy as np

def normalize_socre(adict):
    temperature = 5
    values = np.array(list(adict.values())) / temperature
    softmax_sum = np.sum(np.exp(values - np.max(values)))
    new_dict = {}
    for key, value in adict.items():
        new_dict[key] = np.exp(value / temperature - np.max(values)) / softmax_sum
    return new_dict


print(predict_scores[0])
print(normalize_socre(predict_scores[0]))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# plt.rcParams['font.family'] = 'Times New Roman'

data = []
labels = predict_labels

label_zh_en_map = {
    "顶级": "Top",
    "高级": "High",
    "中级": "Mid",
    "初级": "Low",
    "无关": "NR",
}

ground_truths = [gt if gt < 3 else 3 for gt in ground_truths]

for i in range(len(ground_truths)):
    gt = ground_truths[i]
    normalize_dict = normalize_socre(predict_scores[i])
    for pred in normalize_dict.keys():
        data.append([gt, label_zh_en_map[pred], normalize_dict[pred]])


# 将数据转换为DataFrame
df = pd.DataFrame(data, columns=["ground_truth", "predict_label", "pk"])

# 设置Seaborn的风格
sns.set_theme(style="darkgrid")

# 创建图形
fig, axes = plt.subplots(1, len(set(ground_truths)), figsize=(18, 6))

for i, gt in enumerate(list(set(ground_truths))):
    subset = df[df["ground_truth"] == gt]
    sns.violinplot(
        x="predict_label",
        y="pk",
        data=subset,
        hue="predict_label",
        ax=axes[i],
        order=["NR", "Low", "Mid", "High", "Top"],
        hue_order=["NR", "Low", "Mid", "High", "Top"],
        palette="muted",
        inner="box",
        width=0.8,
        cut=2,
        linewidth=1.25,
    )
    if gt < 3:
        axes[i].set_title(f"Ground-Truth Relevance = {gt}", fontsize=17)
    else:
        axes[i].set_title(f"Ground-Truth Relevance >= 3", fontsize=17)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].set_ylim(-0.1, 1.1)
    axes[i].tick_params(axis='x', labelsize=19)  # x 轴刻度字体大小
    axes[i].tick_params(axis='y', labelsize=17)  # x 轴刻度字体大小
    
axes[0].set_ylabel(r"$p_k$", fontsize=20)
axes[0].set_xlabel(r"(a)", fontsize=20)
axes[1].set_xlabel(r"(b)", fontsize=20)
axes[2].set_xlabel(r"(c)", fontsize=20)
axes[3].set_xlabel(r"(d)", fontsize=20)
plt.tight_layout()
plt.savefig("figs/DrRank_V2.Qwen2.5-7B-Instruct.criteria.L5.pdf")