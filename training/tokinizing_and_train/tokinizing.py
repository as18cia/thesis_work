# https://huggingface.co/distilbert-base-cased-distilled-squad
# https://huggingface.co/deepset/roberta-base-squad2
# https://huggingface.co/deepset/minilm-uncased-squad2

import json

import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from pathlib import Path


def load_data():
    data = json.loads(Path("../preprocessing/data_set_split_how.json").read_bytes())
    # removing items with bad start/end position for answers
    to_remove = []
    for k, v in data.items():
        for i, item in enumerate(v):
            if pd.isna(item[3]) or pd.isna(item[4]):
                to_remove.append((k, i))

    to_remove_training = [x[1] for x in to_remove if x[0] == "training_set"]
    to_remove_eval = [x[1] for x in to_remove if x[0] == "evaluation_set"]

    data["training_set"] = [item for i, item in enumerate(data["training_set"]) if i not in to_remove_training]
    data["evaluation_set"] = [item for i, item in enumerate(data["evaluation_set"]) if i not in to_remove_eval]

    training_data, evaluation_data = data["training_set"], data["evaluation_set"]
    train_contexts, train_questions, train_answers = [x[0] for x in training_data], [x[1] for x in training_data], [
        x[2:] for x in training_data]
    eval_contexts, eval_questions, eval_answers = [x[0] for x in evaluation_data], [x[1] for x in evaluation_data], [
        x[2:] for x in evaluation_data]

    modified_train_answers = []
    for item in train_answers:
        modified_train_answers.append({
            "answer": item[0],
            "answer_start": int(item[1]),
            "answer_end": int(item[2])
        })
    modified_eval_answers = []
    for item in eval_answers:
        modified_eval_answers.append({
            "answer": item[0],
            "answer_start": int(item[1]),
            "answer_end": int(item[2])
        })

    train_answers = modified_train_answers
    eval_answers = modified_eval_answers

    x = 10
    return train_contexts[:x], train_questions[:x], train_answers[:x], eval_contexts[:x], eval_questions[:x], eval_answers[:x]


def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []

    # todo: do we have None for start and end index
    count = 0
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, int(answers[i]['answer_start'])))
        end_positions.append(encodings.char_to_token(i, int(answers[i]['answer_end'])))

        # if start position is None, the answer passage has been truncated
        # todo: how many abstracts have been truncated
        if start_positions[-1] is None:
            count += 1
            start_positions[-1] = tokenizer.model_max_length

        # if end position is None, the 'char_to_token' function points to the space after the correct token, so add - 1
        if end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - 1)
            # if end position is still None the answer passage has been truncated
            if end_positions[-1] is None:
                count += 1
                end_positions[-1] = tokenizer.model_max_length

    # print(count)
    # todo: what is this doing ?
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


def tokenize():
    train_contexts, train_questions, train_answers, eval_contexts, eval_questions, eval_answers = load_data()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")

    # todo: what should be done about truncation
    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(eval_contexts, eval_questions, truncation=True, padding=True)

    add_token_positions(train_encodings, train_answers, tokenizer)
    add_token_positions(val_encodings, eval_answers, tokenizer)

    return train_encodings, val_encodings


if __name__ == '__main__':
    tokenize()
