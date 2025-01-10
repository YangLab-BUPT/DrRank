import json
import numpy as np
import pytrec_eval
import random


def aggregate_score(scores, label_map, strategy="mean"):
    """根据策略聚合分数"""
    def _get_score_by_mean(adict):
        score = 0
        temperature = 1
        values = np.array(list(adict.values())) / temperature
        softmax_sum = np.sum(np.exp(values - np.max(values)))
        for key, value in adict.items():
            score += (
                    np.exp(value / temperature - np.max(values))
                    * label_map[key]
                    / softmax_sum
            )
        return score

    def _get_score_by_max(adict):
        max_key = max(adict, key=adict.get)
        return label_map[max_key]

    aggregated_scores = []
    for query_scores in scores:
        query_aggregated = []
        for pos_score_dict in query_scores:
            if strategy == "mean":
                query_aggregated.append(_get_score_by_mean(pos_score_dict))
            elif strategy == "max":
                query_aggregated.append(_get_score_by_max(pos_score_dict))
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
        aggregated_scores.append(query_aggregated)

    return aggregated_scores


def split_into_evaluation_groups(pos_scores, aggregated_scores, group_size):
    """将一个 query 的 pos 分成多个评估组。"""
    # num_groups = int(np.ceil(len(pos_scores) / group_size))
    num_groups = 1000
    groups = []
    indices = list(range(len(pos_scores)))

    for i in range(num_groups):
        random.seed(i + 42)
        sampled_indices = random.sample(indices, min(group_size, len(indices)))
        # indices = [idx for idx in indices if idx not in sampled_indices]
        group = {
            "true_scores": [pos_scores[i] for i in sampled_indices],
            "pred_scores": [aggregated_scores[i] for i in sampled_indices],
            "indices": sampled_indices
        }
        groups.append(group)
    return groups, num_groups


def calculate_metrics_per_group(group, k_values):
    true_scores = group["true_scores"]
    pred_scores = group["pred_scores"]

    # 构造 pytrec_eval 格式的 ground truth 和预测结果
    ground_truth = {str(idx): score for idx, score in enumerate(true_scores)}
    predictions = {str(idx): float(pred_scores[idx]) for idx in range(len(pred_scores))}

    # 定义 pytrec_eval 的指标
    metric_keys = [
        f"map_cut.{','.join(map(str, k_values))}",
        f"ndcg_cut.{','.join(map(str, k_values))}",
        f"recall.{','.join(map(str, k_values))}",
        f"P.{','.join(map(str, k_values))}"
    ]
    evaluator = pytrec_eval.RelevanceEvaluator(
        {str(0): ground_truth}, set(metric_keys)
    )

    # 执行评估
    scores = evaluator.evaluate({str(0): predictions})["0"]

    # 整理结果
    metrics = {}
    for k in k_values:
        metrics[f"Recall@{k}"] = scores[f"recall_{k}"]
        metrics[f"Precision@{k}"] = scores[f"P_{k}"]
        metrics[f"NDCG@{k}"] = scores[f"ndcg_cut_{k}"]
        metrics[f"MAP@{k}"] = scores[f"map_cut_{k}"]

    def evaluate_mrr(predictions, ground_truth, cutoffs):
        """
        Evaluate MRR for the given predictions and ground truth.
        """
        metrics = {}
        mrrs = np.zeros(len(cutoffs))
        sorted_indices = np.argsort(list(predictions.values()))[::-1]
        for k, cutoff in enumerate(cutoffs):
            for i, idx in enumerate(sorted_indices[:cutoff], 1):
                if ground_truth[str(idx)] > 0:
                    mrrs[k] += 1 / i
                    break
        mrrs /= 1  # 这里默认只有一组数据，不需要取平均
        for i, cutoff in enumerate(cutoffs):
            metrics[f"MRR@{cutoff}"] = round(mrrs[i], 5)
        return metrics

    mrr_scores = evaluate_mrr(predictions, ground_truth, k_values)
    metrics.update(mrr_scores)

    def calculate_pnr(labels, scores):
        """
        Calculate Positive-Negative Ratio (PNR) for a single example.
        """
        assert len(labels) == len(scores)
        nums = len(labels)
        if nums <= 1:
            return -1

        # Create a list of tuples (label, score) and sort by label in descending order
        sorted_items = sorted(zip(labels, scores), key=lambda x: -x[0])
        pos = 0
        neg = 0
        for i in range(nums):
            for j in range(i + 1, nums):
                if sorted_items[i][0] == sorted_items[j][0]:
                    continue
                if sorted_items[i][1] > sorted_items[j][1]:
                    pos += 1
                elif sorted_items[i][1] < sorted_items[j][1]:
                    neg += 1

        if neg == 0:
            return 666666  # Avoid division by zero
        return round(pos / neg, 2)

    # 对 group 的 PNR 进行计算
    pnr = calculate_pnr(true_scores, pred_scores)
    metrics["PNR"] = pnr

    return metrics


def evaluate(data, label_map, k_values, group_size, score_strategy="mean"):
    # 计算聚合分数
    if isinstance(data[0]["scores"][0], dict):
        aggregated_scores = aggregate_score(
            [entry["scores"] for entry in data], label_map, strategy=score_strategy
        )
    else:
        aggregated_scores = [entry["scores"] for entry in data]
    true_pos_scores = [entry["pos_scores"] for entry in data]

    # 初始化所有指标的容器
    mrr_metrics = {f"MRR@{k}": 0 for k in k_values}
    recall_metrics = {f"Recall@{k}": 0 for k in k_values}
    ndcg_metrics = {f"NDCG@{k}": 0 for k in k_values}
    map_metrics = {f"MAP@{k}": 0 for k in k_values}
    precision_metrics = {f"Precision@{k}": 0 for k in k_values}
    pnr_total = 0
    valid_pnr_groups = 0  # 记录有效PNR组的数量
    total_groups = 0

    # 遍历每个查询，分组计算指标
    for query_index, (query_true_scores, query_aggregated_scores) in enumerate(
            zip(true_pos_scores, aggregated_scores)
    ):
        # 创建评估组
        groups, num_groups = split_into_evaluation_groups(
            query_true_scores, query_aggregated_scores, group_size
        )
        total_groups += num_groups

        # 计算每个组的指标
        for group in groups:
            group_metrics = calculate_metrics_per_group(group, k_values)

            # 累加指标
            for k in k_values:
                mrr_metrics[f"MRR@{k}"] += group_metrics[f"MRR@{k}"]
                recall_metrics[f"Recall@{k}"] += group_metrics[f"Recall@{k}"]
                ndcg_metrics[f"NDCG@{k}"] += group_metrics[f"NDCG@{k}"]
                map_metrics[f"MAP@{k}"] += group_metrics[f"MAP@{k}"]
                precision_metrics[f"Precision@{k}"] += group_metrics[f"Precision@{k}"]

            # 检查 PNR 是否异常
            if group_metrics["PNR"] != 666666:  # 排除异常值
                pnr_total += group_metrics["PNR"]
                valid_pnr_groups += 1  # 仅统计有效的PNR组

    # 平均化所有指标
    for k in k_values:
        mrr_metrics[f"MRR@{k}"] = round(mrr_metrics[f"MRR@{k}"] / total_groups, 5)
        recall_metrics[f"Recall@{k}"] = round(recall_metrics[f"Recall@{k}"] / total_groups, 5)
        ndcg_metrics[f"NDCG@{k}"] = round(ndcg_metrics[f"NDCG@{k}"] / total_groups, 5)
        map_metrics[f"MAP@{k}"] = round(map_metrics[f"MAP@{k}"] / total_groups, 5)
        precision_metrics[f"Precision@{k}"] = round(precision_metrics[f"Precision@{k}"] / total_groups, 5)

    # 计算有效PNR的平均值
    if valid_pnr_groups > 0:
        pnr_avg = pnr_total / valid_pnr_groups
    else:
        pnr_avg = 0  # 如果没有有效PNR值，则设为0或其他默认值

    # 打印最终结果
    # print(f"{'-' * 10} Evaluation {'-' * 10}")
    # print(mrr_metrics)
    # print(recall_metrics)
    # print(ndcg_metrics)
    # print(map_metrics)
    # print(precision_metrics)
    # print(f"NDCG@10: {ndcg_metrics['NDCG@10']*100:.2f}")
    # print(f"PNR: {round(pnr_avg, 2)}")
    return round(ndcg_metrics['NDCG@10']*100, 2), round(pnr_avg, 2)

def _calculate_average_time(time_list):
        if len(time_list) == 0:
            return "N/A"
        # 去掉 2 个最低分，2 个最高分，取平均值
        time_list = sorted(time_list)[2:-2]
        return round(np.mean(time_list), 2)

def get_label_map(label_nums):
    assert 2 <= label_nums <= 5
    label_map = None
    if label_nums == 2:
        label_map = {"高级": 1, "无关": 0}
    elif label_nums == 3:
        label_map = {"高级": 2, "初级": 1, "无关": 0}
    elif label_nums == 4:
        label_map = {"高级": 3, "中级": 2, "初级": 1, "无关": 0}
    elif label_nums == 5:
        label_map = {"顶级": 4, "高级": 3, "中级": 2, "初级": 1, "无关": 0}

    # label_map["无关"] = -1
    return label_map 

if __name__ == "__main__":
    import re

    label_map = get_label_map(5)
    k_values = [1, 5, 10, 15, 20]  # 评估指标的 top-k
    group_size = 50  # 每个评估组的样本数量
    
    iters = ["7B", "14B", "32B"]
    # iters = [2,3,4,5]
    # iters = [".female", ".male", ".rural", ".urban"]
    
    # fixed
    size = "7B"
    is_criteria = True
    num = 5
    prefix = ""
    
    score_strategy = "mean" # "mean" or "max"
    
    
    criteria_flag = ".criteria" if is_criteria else ""
    for size in iters:
        input_path = f"runs/DrRank_V2.Qwen2.5-{size}-Instruct{criteria_flag}.L{num}{prefix}.jsonl"
        log_path = f"logs/DrRank.Qwen2.5-{size}-Instruct{criteria_flag}.L{num}.log"
        
        # 读取数据
        with open(input_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        with open(log_path, "r", encoding="utf-8") as f:
            log_data = "".join(f.readlines())
            matches = re.findall(r"The scoring time of every batch is:  \[(.*?)\]", log_data)
            if len(matches) == 0 or len(matches) > 1:
                print(f"Error score_time: {log_path}")
                continue
            score_time = [float(x) for x in matches[0].split(",")]
            
            matches = re.findall(r"The explaining time of every batch is:  \[(.*?)\]\n?", log_data)
            if len(matches) == 0 or len(matches) > 1:
                print(f"Error explanation_time: {log_path}")
                continue
            explanation_time = [float(x) for x in matches[0].split(",")]
            assert len(score_time) == len(explanation_time)
            # print(f"len score_time and explanation_time: {len(score_time)}")
            
            matches = re.findall(r"batch_size=(\d+),", log_data)
            if len(matches) != 1:
                print(f"Error batch_size: {log_path}, len(matches)={len(matches)}")
                continue
            batch_size = int(matches[0])

            # 运行评估
            ndcg_10, pnr = evaluate(data, label_map, k_values, group_size, score_strategy=score_strategy)
            score_time = _calculate_average_time(score_time)
            explanation_time = _calculate_average_time(explanation_time)
            
            print(f"{'-' * 10} Evaluation {'-' * 10}")
            print(f"Evaluating {input_path}")
            print("batch_size, ndcg_10, pnr, score_time, explanation_time")
            alist = [batch_size, ndcg_10, pnr, score_time, explanation_time]
            print(" & ".join(map(str, alist)))
        

