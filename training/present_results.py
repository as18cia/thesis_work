import json
from glob import glob

import pandas as pd
import plotly.graph_objects as go
from tabulate import tabulate


class ResultPresenter:

    def __init__(self):
        self.model_names = ["distilbert-base-cased-distilled-squad", "deepset/roberta-base-squad2",
                            "deepset/minilm-uncased-squad2"]
        self.questions = ["", "what", "how", "which"]
        self.metrics_path = "../data/results/base_results/"
        self.learning_rates = [0.00005, 0.0001]

    def present(self):
        for lr in self.learning_rates:
            accuracy = []
            containment = []
            f1_exact = []
            f1_containment = []

            for m in self.model_names:
                m = m.replace("/", "_")
                acc_res = []
                containment_res = []
                f1_exact_re = []
                f1_containment_re = []
                for q in self.questions:
                    path = f"../data/results/trained_models_results/{m}/_{q}/{str(lr)}/*"
                    target_path = glob(path, recursive=False)
                    assert len(target_path) == 1
                    path = target_path[0] + "/eval_results/metrics.json"

                    with open(path, "r") as f:
                        data = json.load(f)
                        acc_res.append('{:.1f}'.format(float(str(data["accuracy"])[:4])))
                        containment_res.append('{:.1f}'.format(float(str(data["containment"])[:4])))

                        f1_exact_re.append('{:.1f}'.format(float(str(data["f1_score_exact"])[:4])))
                        f1_containment_re.append('{:.1f}'.format(float(str(data["f1_score_containment"])[:4])))

                accuracy.append(acc_res)
                containment.append(containment_res)
                f1_exact.append(f1_exact_re)
                f1_containment.append(f1_containment_re)

            for metric in [(accuracy, "accuracy"), (containment, "containment"), (f1_exact, "f1_exact"),
                           (f1_containment, "f1_containment")]:
                fig = go.Figure(data=[go.Table(
                    header=dict(values=["question", self.model_names[0], self.model_names[1], self.model_names[2]]),
                    cells=dict(values=[self.questions] + [x for x in metric[0]]))
                ], layout={"title": metric[1] + " " + str(lr)})
                fig.show()

    # todo: move me somewhere else maybe ?
    def length_of_predictions(self):
        results = []
        for model in self.model_names:
            model = model.replace("/", "_")
            for lr in self.learning_rates:
                for question in self.questions:
                    path = f"../data/results/trained_models_results/{model}/_{question}/{str(lr)}/*"
                    target_path = glob(path, recursive=False)
                    assert len(target_path) == 1
                    path = target_path[0] + "/eval_results/eval.csv"
                    df = pd.read_csv(path)
                    predication = df["predict_answer"].tolist()
                    number_of_tokens_predicted = sum([len(str(item).strip().split(" ")) for item in predication])
                    results.append((model, lr, question, '{:.1f}'.format(number_of_tokens_predicted / len(df))))
        print("trained models")
        print(tabulate(results, headers=['model', 'lr', 'questions', 'tokens_predicted_per_question']))

        print("base models")
        results = []
        for model in self.model_names:
            model = model.replace("/", "_")
            for lr in self.learning_rates:
                for question in  ["_", "what", "how", "which"]:
                    path = f"../data/results/base_models_results/{model}/{question}/eval.csv"
                    df = pd.read_csv(path)
                    predication = df["predict_answer"].tolist()
                    number_of_tokens_predicted = sum([len(str(item).strip().split(" ")) for item in predication])
                    results.append((model, None, question, '{:.1f}'.format(number_of_tokens_predicted / len(df))))

        print(tabulate(results, headers=['model', 'lr', 'questions', 'tokens_predicted_per_question']))


if __name__ == '__main__':
    # Note: you might need to rerun a few times until free ports are found
    r = ResultPresenter()
    # r.present()
    r.length_of_predictions()
