import json
from glob import glob

import plotly.graph_objects as go


class ResultPresenter:

    def __init__(self):
        self.model_names = ["distilbert-base-cased-distilled-squad", "deepset/roberta-base-squad2",
                            "deepset/minilm-uncased-squad2"]
        self.questions = ["_", "what", "how", "which"]
        self.learning_rates = [0.00005, 0.0001]

    def present(self):

        # for lr in self.learning_rates:
        accuracy = []
        containment = []
        f1_exact = []
        f1_containment = []

        for m in self.model_names:
            acc_res = []
            containment_res = []
            f1_exact_re = []
            f1_containment_re = []
            for q in self.questions:
                # path = "./results/" + m + "/_" + q + "/" + str(lr) + "/*"
                path = "./base_results/{}/{}".format(m, q)
                # target_path = glob(path, recursive=False)
                # assert len(target_path) == 1
                path = path + "/metrics.json"

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
            ], layout={"title": metric[1] + " base"})
            fig.show()



if __name__ == '__main__':
    # app.run_server(port=3003)
    r = ResultPresenter()
    r.present()
