import re
import string


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s)))


def exact_match_score(prediction, ground_truth):
    flag = False  # whether with options
    Choice = ['A', 'B', 'C', 'D']
    for char in normalize_answer(ground_truth):
        if char not in Choice:
            flag = True
            break
    res = 0
    if not flag:
        if normalize_answer(prediction) == normalize_answer(ground_truth):
            res = 1
        elif set(normalize_answer(prediction)).issubset(set(normalize_answer(ground_truth))):
            res = 0.25  # has many correct options
    else:
        try:
            pre = float(prediction)
            gt = float(ground_truth)
            res = int(pre == gt)
        except ValueError:
            if ground_truth.lower().replace(" ", "") in prediction.lower().replace(" ", ""):
                res = 1

    print(prediction, ground_truth, f"| score={res}")
    print("=" * 20)
    return res


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# def compute_exact_match(predictions, references,file_name):
#     exact_match = 0
#     correct = 0
#     half_correct = 0
#     # with open(os.environ["LEVAL_PROMPT_LENGTHS"], 'r', encoding='utf-8') as f:
#     #     data_len = json.load(f)
#     data_name = file_name.split('/')[-1].split('.')[0]
#     save_dir = os.path.join("Predictions", "exam_eval", file_name.split('/')[2], "result.json")
#     len_dist = data_len[data_name]
    
#     for prediction, ground_truths in zip(predictions, references):
#         res = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
#         exact_match += res
#         if res == 1:
#             correct += 1
#         if res == 0.25:
#             half_correct += 1
#     print(
#         f"There are {correct} correct answers \n [for coursera:] {half_correct} can not select all correct options\n Total: {len(predictions)} questions.")
#     print(data_name,save_dir)
#     return 100.0 * exact_match / len(predictions)


def compute_exact_match(predictions, references, file_name):
    import json
    import os
    
    exact_match = 0
    correct = 0
    half_correct = 0
    
    # Initialize counters for each length range
    range_0_4k = {"count": 0, "exact_match": 0}
    range_4k_8k = {"count": 0, "exact_match": 0}
    range_8k_plus = {"count": 0, "exact_match": 0}
    
    prompt_lengths_path = os.environ.get(
        "LEVAL_PROMPT_LENGTHS",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "file_prompt_lengths.json"),
    )
    with open(prompt_lengths_path, 'r', encoding='utf-8') as f:
        data_len = json.load(f)
    
    data_name = file_name.split('/')[-1].split('.')[0]
    leval_dir = os.path.dirname(os.path.dirname(__file__))
    save_dir = os.path.join(leval_dir, "Predictions", "exam_eval", file_name.split('/')[2], "result.json")
    len_dist = data_len[data_name]
    
    for idx, (prediction, ground_truths) in enumerate(zip(predictions, references)):
        res = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        exact_match += res
        
        # Categorize by length range
        doc_length = len_dist[idx]
        if doc_length <= 4096:  # 0-4k
            range_0_4k["count"] += 1
            range_0_4k["exact_match"] += res
        elif doc_length <= 8192:  # 4k-8k
            range_4k_8k["count"] += 1
            range_4k_8k["exact_match"] += res
        else:  # 8k+
            range_8k_plus["count"] += 1
            range_8k_plus["exact_match"] += res
        
        if res == 1:
            correct += 1
        if res == 0.25:
            half_correct += 1
    
    # Calculate scores for each range
    range_0_4k_score = 100.0 * range_0_4k["exact_match"] / range_0_4k["count"] if range_0_4k["count"] > 0 else 0
    range_4k_8k_score = 100.0 * range_4k_8k["exact_match"] / range_4k_8k["count"] if range_4k_8k["count"] > 0 else 0
    range_8k_plus_score = 100.0 * range_8k_plus["exact_match"] / range_8k_plus["count"] if range_8k_plus["count"] > 0 else 0
    overall_exact_match = 100.0 * exact_match / len(predictions) if len(predictions) > 0 else 0
    
    print(f"There are {correct} correct answers \n [for coursera:] {half_correct} can not select all correct options\n Total: {len(predictions)} questions.")
    print(f"Exact Match by length range:")
    print(f"0-4k: {range_0_4k_score:.2f}% ({range_0_4k['count']} samples)")
    print(f"4k-8k: {range_4k_8k_score:.2f}% ({range_4k_8k['count']} samples)")
    print(f"8k+: {range_8k_plus_score:.2f}% ({range_8k_plus['count']} samples)")
    print(f"Overall: {overall_exact_match:.2f}%")
    print(data_name, save_dir)
    
    # Save range statistics to file
    try:
        # First check if the file exists and load its content
        result_data = {}
        
        if os.path.exists(save_dir):
            with open(save_dir, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
        
        # Calculate new scores based on existing data if present
        new_scores = {
            "0-4k": round(range_0_4k_score, 2),
            "4-8k": round(range_4k_8k_score, 2),
            "8k+": round(range_8k_plus_score, 2)
        }
        
        # If data_name already exists in the results, average the scores
        if data_name in result_data:
            existing_scores = result_data[data_name]
            result_data[data_name] = {
                "0-4k": round((existing_scores["0-4k"] + new_scores["0-4k"]) / 2, 2),
                "4-8k": round((existing_scores["4-8k"] + new_scores["4-8k"]) / 2, 2),
                "8k+": round((existing_scores["8k+"] + new_scores["8k+"]) / 2, 2)
            }
        else:
            # Otherwise, just use the new scores
            result_data[data_name] = new_scores
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        
        # Save the updated results
        with open(save_dir, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4)
            
        print(f"Range statistics saved to {save_dir}")
    except Exception as e:
        print(f"Error saving range statistics: {e}")
    
    # Return the original value as before
    return 100.0 * exact_match / len(predictions)
