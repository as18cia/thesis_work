# https://huggingface.co/distilbert-base-cased-distilled-squad
# https://huggingface.co/deepset/roberta-base-squad2
# https://huggingface.co/deepset/minilm-uncased-squad2
import re
from copy import deepcopy
import pandas as pd
from pandas import DataFrame

import path_creator
from collections import Counter
from random import shuffle
import json


# tokenizer = AutoTokenizer.from_pretrained('roberta-base')
# encoded_sentence = tokenizer("weird stuf here")
# print(tokenizer.decode(encoded_sentence["input_ids"]))


def split_training_and_evaluation():
    df = pd.read_csv("./data_set_raw.csv")
    # df["start_ind"] = None
    # df["end_ind"] = None

    #  todo: is there some issue with the data
    #  todo: < 10 -> training not evaluation
    to_delete = []
    for i, data in df.iterrows():
        try:
            if not find_word_in_text(data["object_label"])(data["paper_abstract"]):
                to_delete.append(i)
                continue
        except:
            pass

        try:
            # answer index position
            start, end = find_word_in_text(data["object_label"])(data["paper_abstract"]).span()
            df.at[i, "start_ind"] = start
            df.at[i, "end_ind"] = end
        except:
            pass

    # dropping bad data
    df.drop(to_delete, inplace=True)
    predicates = df["predicate_label"].tolist()
    counts = Counter(predicates)

    # splitting the data
    training_set = []
    evaluation_set = []
    for p, count in counts.items():
        if count < 10:
            data = [x.tolist() for _, x in (df[df["predicate_label"] == p].iterrows())]
            training_set.extend(data)
        else:
            data = [x.tolist() for _, x in (df[df["predicate_label"] == p].iterrows())]
            shuffle(data)
            training_set.extend(data[:round(0.8 * len(data))])
            evaluation_set.extend(data[round(0.8 * len(data)):])

    data_set = {
        "training_set": training_set,
        "evaluation_set": evaluation_set,
    }

    # as of last check 4515 703
    print(len(data_set["training_set"]), len(data_set["evaluation_set"]))

    def add_question(y, q):
        x = deepcopy(y)
        x[1] = q + " " + x[1] + " ?"
        return x

    for q in ["what", "which", "how", ""]:
        data_set_m = {
            "training_set": [add_question(x, q) for x in training_set],
            "evaluation_set": [add_question(x, q) for x in evaluation_set],
        }
        with open(f"./data_set_split_{q}.json", mode="w") as file:
            json.dump(data_set_m, file)

    print(len(training_set), len(evaluation_set))


def create_training_dataset():
    df = pd.read_csv(path_creator.path("data_set.csv"))
    df = df[["paper_abstract", "predicate_label", "object_label"]]
    df.to_csv("./training_dataset.csv", index=False)


def find_word_in_text(w):
    return re.compile(r'\b({0})\b'.format(r'{}'.format(w)), flags=re.IGNORECASE).search


if __name__ == '__main__':
    pass

