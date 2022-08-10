import datetime
import json
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, IntervalStrategy, Trainer
from training.tokinizing_and_train.DataSet import MyDataset
from training.tokinizing_and_train.tokinizing import tokenize
from transformers import default_data_collator


class TrainAndEvaluation:

    def __init__(self, model_name: str, question_label: str, cased_model: bool, learning_rate: float, epochs: int,
                 training_test_len=None, eval_test_len=None):

        self.result_model_path = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                   else 'cpu')

        self.cased_model = cased_model
        # self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # train_encodings, val_encodings = tokenize(self.tokenizer, question_label, cased_model, training_test_len,
        #                                           eval_test_len)

        # self.train_set = MyDataset(train_encodings)
        # self.val_set = MyDataset(val_encodings)

        self.learning_rate = learning_rate  # 0.01  # 0.0001
        self.epochs = epochs

        self.data_collator = default_data_collator
        self.model_name = model_name
        self.question_label = question_label

    def train(self):
        # whole_train_eval_time = time.time()

        training_args = TrainingArguments(
            output_dir='../results',
            evaluation_strategy=IntervalStrategy.EPOCH,
            save_strategy=IntervalStrategy.EPOCH,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            load_best_model_at_end=True

        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_set,
            eval_dataset=self.val_set,
            data_collator=default_data_collator,
            tokenizer=self.tokenizer,
        )

        trainer.train()

        # creating path to save the trained model
        local_time = datetime.datetime.now()
        folder = "/content/drive/MyDrive/train_in_colab/results/{}/{}/{}/{}".format(self.model_name,
                                                                                    "_" + self.question_label,
                                                                                    str(self.learning_rate),
                                                                                    local_time.__str__().replace(":",
                                                                                                                 "."))
        Path(folder).mkdir(parents=True, exist_ok=True)
        self.result_model_path = folder

        trainer.save_model(folder)

    def evaluate(self, model_path: str, data_path):
        # path = self.result_model_path
        path = model_path
        model = AutoModelForQuestionAnswering.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)

        eval_data = self._load_data_set(data_path)["evaluation_set"]

        results = []
        success = 0
        containment = 0
        no_answer = 0

        total_tokens_predicted = 0
        total_correct_tokens_predicted_exact = 0
        total_correct_tokens_predicted_containment = 0
        total_tokens_expected = 0

        for item in tqdm(eval_data, total=len(eval_data)):
            if self.cased_model:
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

        recall_exact, precision_exact, f1_exact = self._f1_score(total_tokens_expected, total_tokens_predicted,
                                                                 total_correct_tokens_predicted_exact)

        recall_containment, precision_containment, f1_containment = self._f1_score(total_tokens_expected,
                                                                                   total_tokens_predicted,
                                                                                   total_correct_tokens_predicted_containment)

        csv_path = path + '/eval_results/'
        Path(csv_path).mkdir(parents=True, exist_ok=True)
        csv_path = csv_path + "eval.csv"
        pd.DataFrame(results, columns=["abstract", "question", "answer", "predict_answer", "exact_match",
                                       "containment_match"]).to_csv(csv_path,
                                                                    index=False)

        metric_path = path + "/eval_results/"
        Path(metric_path).mkdir(parents=True, exist_ok=True)
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

        with open(metric_path + "metrics.json", "w") as file:
            json.dump(res_dic, file)

    def _load_data_set(self, data_path):
        return json.loads(Path(
            "{}/data_set_split_{}.json".format(data_path, self.question_label)).read_bytes())

    @staticmethod
    def _f1_score(expected_tokens, predicted_tokens, correct_predicted_tokens):
        if correct_predicted_tokens == 0:
            return 0

        recall = correct_predicted_tokens / expected_tokens
        precision = correct_predicted_tokens / predicted_tokens
        f1 = (2 * recall * precision) / (recall + precision)
        return recall, precision, f1


if __name__ == '__main__':
    from glob import glob

    for model, c in [("deepset_minilm-uncased-squad2", False),
                     ("deepset_roberta-base-squad2", True)]:
        for label in ["", "what", "which", "how"]:
                tr = TrainAndEvaluation(model_name=model, question_label=label,
                                        cased_model=c,
                                        learning_rate=lr, epochs=4)

                path = "./results/{}/{}/{}/*".format(model, "_" + label, str(lr))
                target_path = glob(path, recursive=False)
                assert len(target_path) == 1
                target_path = target_path[0]

                data_path = "../preprocessing"
                tr.evaluate(target_path, data_path)

# todo: clean code and upload to orkg
# todo: upload models to huggingface\
# todo: data analysis -> len of abstracts -> percentages of training vs eval data ... and
# truncation thing


# todo: provide stats:
    # orkg data -> check beginning
    # training data and stuff
    # scores and categories stats

# todo: naming the models