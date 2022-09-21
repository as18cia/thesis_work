import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


def evaluate(model_path: str, label, cased: bool):
    model_name = model_path
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    eval_data = _load_data_set(label)["evaluation_set"]

    results = []
    success = 0
    containment = 0
    no_answer = 0

    total_tokens_predicted = 0
    total_correct_tokens_predicted_exact = 0
    total_correct_tokens_predicted_containment = 0
    total_tokens_expected = 0

    for item in tqdm(eval_data, total=len(eval_data)):
        if cased:
            text = item[0]
            question = item[1].strip().capitalize()
        else:
            text = item[0].lower()
            question = item[1].strip().lower()

        target_answer = item[2].strip().lower()
        total_tokens_expected += len(target_answer.split(" "))

        inputs = tokenizer(question, text, return_tensors="pt", truncation=True, padding=True, max_length=510)
        with torch.no_grad():
            outputs = model(**inputs)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
        predict_answer = tokenizer.decode(predict_answer_tokens)

        # the predicted answer
        predict_answer = predict_answer.strip().lower()

        if not predict_answer:
            no_answer += 1

        containment_match = target_answer in predict_answer
        if containment_match:
            containment += 1
            total_correct_tokens_predicted_containment += len(target_answer.split(" "))

        cleaned_predict_answer = predict_answer

        if cleaned_predict_answer:
            if cleaned_predict_answer[-1] in [".", ",", ";", ":", "-", ")", "(", "_", "+"]:
                cleaned_predict_answer = cleaned_predict_answer[:-1].strip()
                total_tokens_predicted += len(cleaned_predict_answer.split(" "))

        exact_match = cleaned_predict_answer == target_answer
        if exact_match:
            success += 1
            total_correct_tokens_predicted_exact += len(target_answer.split(" "))

        results.append((item[0], item[1], item[2], predict_answer, exact_match, containment_match))

    recall_exact, precision_exact, f1_exact = _f1_score(total_tokens_expected, total_tokens_predicted,
                                                        total_correct_tokens_predicted_exact)

    recall_containment, precision_containment, f1_containment = _f1_score(total_tokens_expected,
                                                                          total_tokens_predicted,
                                                                          total_correct_tokens_predicted_containment)

    saving_path = "../data/results/base_models_results/{}/{}".format(model_name.replace("/", "_"), label)
    Path(saving_path).mkdir(parents=True, exist_ok=True)
    csv_path = saving_path + "/eval.csv"
    pd.DataFrame(results, columns=["abstract", "question", "answer", "predict_answer", "exact_match",
                                   "containment_match"]).to_csv(csv_path,
                                                                index=False)

    res_dic = {
        "accuracy": success / len(eval_data),
        "containment": containment / len(eval_data),
        "no answer": no_answer / len(eval_data),

        "recall_exact": recall_exact,
        "precision_exact": precision_exact,
        "f1_score_exact": f1_exact,

        "recall_containment": recall_containment,
        "precision_containment": precision_containment,
        "f1_score_containment": f1_containment
    }

    for k, v in res_dic.items():
        res_dic[k] = float("{:.3f}".format(v)) * 100

    with open(saving_path + "/metrics.json", "w") as file:
        json.dump(res_dic, file)


def _load_data_set(label):
    return json.loads(Path(
        "../data/training_ready_data/train_and_evaluation_set_{}.json".format(label)).read_bytes())


def _f1_score(expected_tokens, predicted_tokens, correct_predicted_tokens):
    if correct_predicted_tokens == 0:
        return 0

    recall = correct_predicted_tokens / expected_tokens
    precision = correct_predicted_tokens / predicted_tokens
    f1 = (2 * recall * precision) / (recall + precision)
    return recall, precision, f1


if __name__ == '__main__':
    for model, c in [("deepset/minilm-uncased-squad2", False),
                     ("deepset/roberta-base-squad2", True), ("distilbert-base-cased-distilled-squad", True)]:
        for label in ["", "what", "which", "how"]:
            evaluate(model, label, c)
