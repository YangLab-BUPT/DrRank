import json
import numpy as np
import pytrec_eval
import argparse

from modeling.model_bert import CrossEncoder
from modeling.model_llm_logit import LlmPointwiseRanker


# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


def replace_illegal_chars(value):
    if isinstance(value, str):
        return "".join(ch if ord(ch) >= 32 or ch in "\t\n\r" else "_" for ch in value)
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Reranker Configuration")

    # Data arguments
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/DrRank.jsonl",
        help="The data path points to a file in JSONL format. Each line contains `query`, `pos`, `pos_label_scores`. "
        "Here, `query` is a string (`str`), `pos` are lists of strings (`List[str]`) and `pos_label_scores` "
        "are lists of integers (`List[int]`).",
    )

    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The model name or path for the reranker.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="The model type for the reranker, it can be llm-decoder, bi-encoder, cross-encoder.",
    )
    parser.add_argument(
        "--using_criteria",
        default=False,
        action="store_true",
        help="Whether to use the criteria.",
    )
    parser.add_argument(
        "--label_nums",
        type=int,
        default=5,
        help="The number of labels from 2 to 5.",
    )
    parser.add_argument(
        "--is_gpt",
        action="store_true",
        default=False,
        help="Whether the model is an api endpoint.",
    )
    parser.add_argument(
        "--max_doctor_length",
        type=int,
        default=1024,
        help="The maximum length of the input doctor profile.",
    )

    # Inference arguments
    parser.add_argument(
        "--batch_size", type=int, default=16, help="The batch size for inference."
    )

    # Evaluation arguments
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["recall", "mrr", "ndcg", "map", "precision", "pnr"],
        help="The evaluation metrics, you can set recall / mrr / ndcg.",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 15, 50, 100],
        help="Present the top-k metrics evaluation.",
    )
    parser.add_argument(
        "--query_prefix",
        type=str,
        default="",
        help="The prefix for the query for fairness evaluation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"{args=}")
    print(f"{'-'*10} Evaluation {'-'*10}")

    data = []
    data_num = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    if len(args.model_name_or_path.split("/")) > 2:
        base_dir = ""
    else:
        base_dir = ""

    if args.model_type == "bi-encoder":
        pass
    elif args.model_type == "cross-encoder":
        reranker = CrossEncoder.from_pretrained(
            model_name_or_path=base_dir + args.model_name_or_path,
            query_format=f"{args.query_prefix}" + "{}",
        )
    elif args.model_type == "llm-logit":
        reranker = LlmPointwiseRanker.from_pretrained(
            model_name_or_path=base_dir + args.model_name_or_path,
            using_criteria=args.using_criteria,
            label_nums=args.label_nums,
            query_format=f"{args.query_prefix}" + "{}",
        )

    true_labels = []
    pairs = []
    scores = []
    explanations = []
    for d in data:
        data_num.append(0)
        passages = []

        if "pos" in d:
            passages.extend(d["pos"])
            for p_socre in d["pos_scores"]:
                true_labels.append(p_socre)

        if "neg" in d:
            passages.extend(d["neg"])
            for n_socre in d["neg_scores"]:
                true_labels.append(n_socre)

        for p in passages:
            pairs.append((d["query"], p))
            data_num[-1] += 1

        if "scores" in d:
            scores.extend(d["scores"])

        if "explanations" in d:
            explanations.extend(d["explanations"])

        assert len(true_labels) == len(pairs)

    print(f"待评估的医生总数为：{len(pairs)}")

    if len(scores) == 0:
        result = reranker.compute_score(
            pairs,
            batch_size=args.batch_size,
            max_length=args.max_doctor_length,
            normalize=False,
        )

        if isinstance(result, tuple):
            scores, explanations = result
        elif isinstance(result, list):
            scores = result
            explanations = [None] * len(scores)

        print(f"{len(scores)=}")
        print(f"{len(explanations)=}")
        # Now add scores and explanations back to the data
        score_idx = 0
        explanation_idx = 0
        for d in data:
            d["scores"] = []
            d["explanations"] = []
            if "pos" in d:
                # Add scores and explanations for the positive passages
                for _ in d["pos"]:
                    d["scores"].append(scores[score_idx])
                    d["explanations"].append(explanations[explanation_idx])
                    score_idx += 1
                    explanation_idx += 1
            if "neg" in d:
                # Add scores and explanations for the negative passages
                for _ in d["neg"]:
                    d["scores"].append(scores[score_idx])
                    d["explanations"].append(explanations[explanation_idx])
                    score_idx += 1
                    explanation_idx += 1
        assert score_idx == len(scores) and explanation_idx == len(explanations)
        if args.using_criteria:
            criteria_flag = ".criteria"
            # criteria_flag = ".random_criteria"
        else:
            criteria_flag = ""
        if args.model_type == "llm-logit":
            label_flag = f".L{args.label_nums}"
        else:
            label_flag = ""
            
        query_prefix_flag = ""
        if args.query_prefix != "{}":
            if "男性" in args.query_prefix:
                query_prefix_flag = ".male"
            elif "女性" in args.query_prefix:
                query_prefix_flag = ".female"    
            elif "城镇" in args.query_prefix:
                query_prefix_flag = ".urban"
            elif "乡村" in args.query_prefix:
                query_prefix_flag = ".rural"
                
        with open(
            args.input_path.replace(
                ".jsonl",
                f".{args.model_name_or_path.split('/')[-1]}{criteria_flag}{label_flag}{query_prefix_flag}.jsonl",
            ).replace("data/", "runs/"),
            "w",
            encoding="utf-8",
        ) as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    else:
        print(f"文件中已有评分，无需再次评分！")

    if isinstance(scores[0], dict):
        # 分值处理策略
        label_map = reranker.label_map
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
            max_key = max(label_map, key=label_map.get)
            # 此时 score 无需 softmax
            score = adict[max_key]
            return score

        scores = list(map(_get_score_by_mean, scores))
        # scores = list(map(_get_score_by_max, scores))

    scores = np.asarray(scores).reshape(-1)

    start_num = 0
    ground_truths = {}
    labels = []
    for i in range(len(data)):
        tmp = {}
        tmp_labels = []
        for ind in range(len(data[i]["pos"])):
            try:
                data[i]["pos_label_scores"] = data[i]["pos_scores"]
                tmp[str(start_num + ind)] = int(data[i]["pos_label_scores"][ind])
            except Exception as e:
                # print(e)
                tmp[str(start_num + ind)] = 1
            if tmp[str(start_num + ind)] == max(true_labels):
                tmp_labels.append(start_num + ind)
        ground_truths[str(i)] = tmp
        start_num += data_num[i]
        labels.append(tmp_labels)

    start_num = 0
    rerank_results = {}
    predicts = []
    for i in range(len(data)):
        tmp = {}
        tmp_predicts = [
            (start_num + ind, scores[start_num + ind]) for ind in range(data_num[i])
        ]
        tmp_predicts = [
            idx for (idx, _) in sorted(tmp_predicts, key=lambda x: x[1], reverse=True)
        ]
        for ind in range(data_num[i]):
            tmp[str(start_num + ind)] = float(scores[start_num + ind])
        rerank_results[str(i)] = tmp
        start_num += data_num[i]
        predicts.append(tmp_predicts)

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in args.k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"Precision@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in args.k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in args.k_values])
    recall_string = "recall." + ",".join([str(k) for k in args.k_values])
    precision_string = "P." + ",".join([str(k) for k in args.k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        ground_truths, {map_string, ndcg_string, recall_string, precision_string}
    )

    scores = evaluator.evaluate(rerank_results)

    for query_id in scores.keys():
        for k in args.k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"Precision@{k}"] += scores[query_id]["P_" + str(k)]

    for k in args.k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"Precision@{k}"] = round(
            precision[f"Precision@{k}"] / len(scores), 5
        )

    def _evaluate_mrr(predicts, labels, cutoffs):
        """
        Evaluate MRR.
        """
        metrics = {}
        mrrs = np.zeros(len(cutoffs))
        for pred, label in zip(predicts, labels):
            jump = False
            for i, x in enumerate(pred, 1):
                if x in label:
                    for k, cutoff in enumerate(cutoffs):
                        if i <= cutoff:
                            mrrs[k] += 1 / i
                    jump = True
                if jump:
                    break
        mrrs /= len(predicts)
        for i, cutoff in enumerate(cutoffs):
            mrr = mrrs[i]
            metrics[f"MRR@{cutoff}"] = mrr
        return metrics

    def _calculate_pnr_single_example(labels, scores):
        assert len(labels) == len(scores)
        nums = len(labels)
        if nums <= 1:
            return -1

        # Create a list of tuples (label, score, index) and sort by label in descending order
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
            return 666666
        # print(f"pos: {pos}, neg: {neg}")
        return round(pos / neg, 2)

    def _calculate_pnr(ground_truths, rerank_results):
        pnr = []
        for key in ground_truths.keys():
            labels = []
            scores = []
            g_dict = ground_truths[key]
            s_dict = rerank_results[key]
            for k in g_dict.keys():
                g = g_dict[k]
                s = s_dict[k]
                labels.append(g)
                scores.append(s)
            _pnr = _calculate_pnr_single_example(labels, scores)
            if _pnr == -1 or _pnr == 666666:
                print(f"Warning PNR: {_pnr}")
                continue
            pnr.append(_pnr)
        return round(np.mean(pnr), 2)

    mrr = _evaluate_mrr(predicts, labels, args.k_values)
    pnr = _calculate_pnr(ground_truths, rerank_results)

    print(f"{'-'*10} Evaluation {'-'*10}")
    print(f"{args.input_path=}")
    print(f"{args.model_name_or_path=}")

    if "mrr" in args.metrics:
        print(mrr)
    if "recall" in args.metrics:
        print(recall)
    if "ndcg" in args.metrics:
        print(ndcg)
    if "map" in args.metrics:
        print(_map)
    if "precision" in args.metrics:
        print(precision)
    if "pnr" in args.metrics:
        print(f"PNR: {pnr}")

    def _calculate_average_time(time_list):
        if len(time_list) == 0:
            return "N/A"
        # 去掉 2 个最低分，2 个最高分，取平均值
        time_list = sorted(time_list)[2:-2]
        return round(np.mean(time_list), 2)
    
    
    print("The scoring time of every batch is: ", reranker.score_time)
    print("The explaining time of every batch is: ", reranker.explanation_time)
    
    print(f"The average scoring time per batch is: {_calculate_average_time(reranker.score_time)}")
    print(f"The average explaining time per batch is: {_calculate_average_time(reranker.explanation_time)}")


if __name__ == "__main__":
    main()
