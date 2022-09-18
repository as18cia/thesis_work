import json
from pathlib import Path
from statistics import mean
import pandas as pd
from tabulate import tabulate


class DataSetStats:

    def __init__(self):
        self.data_set = pd.read_csv("../data/processed/finale_data_set.csv")
        self.split_data_set = json.loads(
            Path("../data/training_ready_data/train_and_evaluation_set_.json").read_bytes())

    def calculate_general_stats(self):
        number_of_rows = len(self.data_set)
        number_of_unique_cats = len(set(self.data_set["ner"].tolist()))
        number_of_unique_predicate = len(set(self.data_set["predicate_label"].tolist()))

        avg_tokens_of_predicate = mean([len(x.split(" ")) for x in self.data_set["predicate_label"].tolist()])
        avg_tokens_of_object = mean([len(x.split(" ")) for x in self.data_set["object_label"].tolist()])
        avg_tokens_of_abstract = mean([len(x.split(" ")) for x in self.data_set["paper_abstract"].tolist()])
        number_of_abs_over_510 = [len(x.split(" ")) for x in self.data_set["paper_abstract"].tolist()]
        number_of_abs_over_510 = len([x for x in number_of_abs_over_510 if x > 510])

        results = []
        results.append(("number_of_rows", number_of_rows))
        results.append(("number_of_unique_categories", number_of_unique_cats))
        results.append(("number_of_unique_predicate_labels", number_of_unique_predicate))
        results.append(("avg_tokens_of_predicate_labels", avg_tokens_of_predicate))
        results.append(("avg_tokens_of_object_labels", avg_tokens_of_object))
        results.append(("avg_tokens_of_abstracts", avg_tokens_of_abstract))
        results.append(("number_of_abstracts_over_510_token(not unique)", number_of_abs_over_510))

        print(tabulate(results, headers=['Measurement', 'Value'], numalign="right"))

    def category_stats(self):
        results = []
        cats = set(self.data_set["ner"].tolist())
        for c in cats:
            percentage_of_data = (self.data_set["ner"].tolist().count(c) / len(self.data_set)) * 100
            percentage_of_data = "{:.2f}".format(percentage_of_data)
            results.append((c, percentage_of_data))

        results.sort(key=lambda x: float(x[1]), reverse=True)
        print(tabulate(results, headers=['category', 'percentage_of_data'], numalign="right"))

    def contribution_distribution(self):
        import matplotlib.pyplot as plt
        import numpy as np
        data = pd.read_csv(r"C:\Users\mhrou\Desktop\ResearchFields_to_Contributions.csv")
        re_fields = [x for x in range(len(data["ResearchFieldLabel"]))][1:]
        number_of_contributions = data["#OfContributions"][1:]


        fig = plt.figure()


        plt.bar(re_fields, number_of_contributions, 0.95, color='b')
        plt.xlabel("research fields (names omitted)")
        plt.ylabel("number of contributions")
        plt.show()

    def place_holder_1(self):
        data = pd.read_csv(r"C:\Users\mhrou\Desktop\merged_statements_to_abstracts_v2.csv")
        print(data.count())
        print(data.nunique())


if __name__ == '__main__':
    d = DataSetStats()
    # d.calculate_general_stats()
    # print("\n        ******** Category stats ********\n")
    # d.category_stats()


    d.contribution_distribution()
