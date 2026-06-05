import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    #parser.add_argument('--exp_name', default='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    
    if args.e:
        path = f"pred_e/{args.model}/"
        csv_path = "pred_e/evaluation_results.csv"
    else:
        path = f"pred/{args.model}/"
        csv_path = "pred/evaluation_results.csv"
    
    all_files = os.listdir(path)
    print("正在评估:", all_files)
    
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        
        scores[dataset] = score
    
    # 保存原始JSON格式
    if args.e:
        out_path = f"pred_e/{args.model}/result.json"
    else:
        out_path = f"pred/{args.model}/result.json"
    
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    
    # 准备CSV数据
    csv_row = {"模型": args.model}
    csv_row.update(scores)
    
    # 计算平均分
    if scores:
        csv_row["平均值"] = round(sum(scores.values()) / len(scores), 4)
    else:
        csv_row["平均值"] = 0
    
    # 定义数据集的固定顺序（使用实际的文件名作为数据集名）
    dataset_order = [
        "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
        "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", 
        "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
    ]
    
    # 检查CSV文件是否存在
    file_exists = os.path.exists(csv_path)
    
    # 写入CSV文件（追加模式）
    import csv
    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
        # 构建字段名列表，按照指定顺序
        fieldnames = ["模型"]
        fieldnames.extend(dataset_order)
        fieldnames.append("平均值")
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 只有文件不存在或为空时才写入表头
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        
        writer.writerow(csv_row)
    
    print(f"结果已保存到 {out_path} 和 {csv_path}")
    print(f"模型 {args.model} 平均分数: {csv_row['平均值']}")