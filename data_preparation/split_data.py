# https://huggingface.co/distilbert-base-cased-distilled-squad
# https://huggingface.co/deepset/roberta-base-squad2
# https://huggingface.co/deepset/minilm-uncased-squad2

import re
from copy import deepcopy
import pandas as pd

import path_creator
from collections import Counter
from random import shuffle
import json


class SplitData:

    def split_training_and_evaluation(self):
        df = pd.read_csv("../data/processed/finale_dataset.csv")
        df = df[["PaperAbstract", "PredicateLabel", "ObjectLabel"]]

        df["StartIndex"] = None
        df["EndIndex"] = None

        for i, data in df.iterrows():
            # answer index position
            try:
                start, end = self.find_word_in_text(data["ObjectLabel"])(data["PaperAbstract"]).span()
                df.at[i, "StartIndex"] = start
                df.at[i, "EndIndex"] = end
            except:
                # todo: further action is required in the case the position couldn't be found
                pass

        predicates = df["PredicateLabel"].tolist()
        counts = Counter(predicates)

        # splitting the data
        training_set = []
        evaluation_set = []
        for p, count in counts.items():
            if count < 10:
                data = [x.tolist() for _, x in (df[df["PredicateLabel"] == p].iterrows())]
                training_set.extend(data)
            else:
                data = [x.tolist() for _, x in (df[df["PredicateLabel"] == p].iterrows())]
                shuffle(data)
                training_set.extend(data[:round(0.74 * len(data))])
                evaluation_set.extend(data[round(0.74 * len(data)):])

        data_set = {
            "training_set": training_set,
            "evaluation_set": evaluation_set,
        }

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
            with open(f"../data/training_ready_data/train_and_evaluation_set_{q}.json", mode="w") as file:
                json.dump(data_set_m, file)

    @staticmethod
    def find_word_in_text(w):
        return re.compile(r'\b({0})\b'.format(r'{}'.format(w)), flags=re.IGNORECASE).search
