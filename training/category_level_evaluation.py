import pandas as pd
from tabulate import tabulate

class CategoryLevelEvaluation:

    def __init__(self):
        self.prediction_data: pd.DataFrame = pd.read_csv(
            "../data/results/trained_models_results/deepset_roberta-base-squad2/_/0.0001/2022-08-03 "
            "05.57.04.355474/eval_results/eval.csv")
        self.complete_dataset = pd.DataFrame = pd.read_csv("../data/processed/finale_data_set.csv")

    def merge(self):
        df_all = self.complete_dataset[["paper_abstract", "predicate_label", "object_label", "ner"]]
        df_predicted = self.prediction_data.drop_duplicates()

        for i, data in df_predicted.iterrows():
            df_predicted.at[i, "question"] = data["question"][:-1].strip()

            df_predicted.at[i, "answer"] = data["answer"].strip()
            df_predicted.at[i, "abstract"] = data["abstract"].strip()

        for i, data in df_all.iterrows():
            df_all.at[i, "paper_abstract"] = data["paper_abstract"].strip()
            df_all.at[i, "predicate_label"] = data["predicate_label"].strip()
            df_all.at[i, "object_label"] = data["object_label"].strip()

        merged = df_predicted.merge(df_all, left_on=["abstract", "question", "answer"],
                                    right_on=["paper_abstract", "predicate_label", "object_label"], how="left")
        merged = merged.drop_duplicates()

        # print(df_predicted.count())
        # print("*********")
        # print(merged.count())

        return merged

    def calculate_metrics(self, df):

        results = []
        cats = list(set(df["ner"]))
        for c in cats:
            dft = df[df["ner"] == c]
            results.append((c, len(dft), '{:.1f}'.format(self._calculate_metrics(dft))))

        results.sort(key=lambda x: x[2])
        print(tabulate(results, headers=['category', 'number_of_rows', 'accuracy_of_prediction']))
        print("*****")
        print(len(df[df["exact_match"] == True]) / len(df) * 100)


    @staticmethod
    def _calculate_metrics(df):
        l = len(df)
        exact_l = len(df[df["exact_match"] == True])
        return exact_l / l * 100


if __name__ == '__main__':
    cc = CategoryLevelEvaluation()
    r_df = cc.merge()
    cc.calculate_metrics(r_df)

# todo: are the abstracts unique
# todo: the deduplication should probably be on (abstract, predicate, object) -> report
