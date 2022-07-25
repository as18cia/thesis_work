import datetime
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, IntervalStrategy, Trainer
from training.tokinizing_and_train.DataSet import MyDataset
from training.tokinizing_and_train.tokinizing import tokenize
from transformers import default_data_collator
import numpy as np


class TrainAndEvaluation:

    def __init__(self, model_name: str, question_label: str, training_test_len=None, eval_test_len=None):
        self.result_model_path = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                   else 'cpu')
        # todo: does this need revisiting -> DistilBertForQuestionAnswering
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_encodings, val_encodings = tokenize(self.tokenizer, question_label, training_test_len, eval_test_len)

        self.train_set = MyDataset(train_encodings)
        self.val_set = MyDataset(val_encodings)

        self.learning_rate = 3e-5
        self.epochs = 2

        self.data_collator = default_data_collator
        self.model_name = model_name
        self.question_label = question_label

    def train(self):
        # whole_train_eval_time = time.time()

        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy=IntervalStrategy.EPOCH,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
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
        folder = "./results/{}/{}/{}".format(self.model_name, "_" + self.question_label,
                                             local_time.__str__().replace(":", "."))
        Path(folder).mkdir(parents=True, exist_ok=True)
        self.result_model_path = folder

        trainer.save_model(folder)

    def evaluate(self):
        path = self.result_model_path
        model = AutoModelForQuestionAnswering.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)

        eval_data = self._load_data_set()["evaluation_set"]
        results = []
        success = 0
        for item in tqdm(eval_data, total=len(eval_data)):
            text = item[0]
            question = item[1]

            inputs = tokenizer(question, text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax()

            predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
            predict_answer = tokenizer.decode(predict_answer_tokens)
            match = predict_answer.lower().strip() == item[2].lower().strip()
            results.append((item[0], item[1], item[2], predict_answer, match))
            if match:
                success += 1

        csv_path = path + '/eval_results/'
        Path(csv_path).mkdir(parents=True, exist_ok=True)
        csv_path = csv_path + "eval.csv"
        pd.DataFrame(results, columns=["abstract", "question", "answer", "predict_answer", "match_?"]).to_csv(csv_path,
                                                                                                              index=False)

        metric_path = path + "/eval_results/"
        Path(metric_path).mkdir(parents=True, exist_ok=True)
        res_dic = {
            "accuracy": success / len(eval_data)
        }
        with open(metric_path + "metrics.json", "w") as file:
            json.dump(res_dic, file)

        # how to map prediction_data to other attributes

    def _load_data_set(self):
        return json.loads(Path("../preprocessing/data_set_split_{}.json".format(self.question_label)).read_bytes())


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


if __name__ == '__main__':
    # todo: the eval step during training ?
    # todo: truncation issue
    # todo: adding the F1 score

    for label in ["", "what", "which", "how"]:
        print("started: " + label)
        tr = TrainAndEvaluation("distilbert-base-cased-distilled-squad", label)
        tr.train()
        tr.evaluate()
