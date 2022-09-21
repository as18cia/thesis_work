import datetime
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, IntervalStrategy, Trainer
from transformers import default_data_collator

from training_evaluation.my_data_set import MyDataset
from training_evaluation.my_tokenizer import Tokenizer


class TrainAndEvaluation:

    def __init__(self, model_name: str, question_label: str, cased_model: bool, learning_rate: float, epochs: int,
                 training_test_len=None, eval_test_len=None):

        self.result_model_path = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                   else 'cpu')

        self.cased_model = cased_model
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenize = Tokenizer()
        train_encodings, val_encodings = tokenize.tokenize(self.tokenizer, question_label, cased_model,
                                                           training_test_len,
                                                           eval_test_len)

        self.train_set = MyDataset(train_encodings)
        self.val_set = MyDataset(val_encodings)

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.data_collator = default_data_collator
        self.model_name = model_name
        self.question_label = question_label

    def train(self):

        training_args = TrainingArguments(
            output_dir='../data/results/temp',
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

        folder = "../data/results/trained_models_results/{}/{}/{}/{}".format(self.model_name.replace("/", "_"),
                                                                             "_" + self.question_label,
                                                                             str(self.learning_rate),
                                                                             local_time.__str__().replace(":",
                                                                                                          "."))
        Path(folder).mkdir(parents=True, exist_ok=True)
        self.result_model_path = folder
        trainer.save_model(folder)

    def evaluate(self):
        trained_model_path = self.result_model_path
        trained_model = AutoModelForQuestionAnswering.from_pretrained(trained_model_path)
        tokenizer = AutoTokenizer.from_pretrained(trained_model_path)

        eval_data = self._load_data_set()["evaluation_set"]

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
                outputs = trained_model(**inputs)
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

        result_path = trained_model_path + '/eval_results/'
        Path(result_path).mkdir(parents=True, exist_ok=True)
        csv_path = result_path + "eval.csv"
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

        with open(result_path + "metrics.json", "w") as file:
            json.dump(res_dic, file)

    def _load_data_set(self):
        return json.loads(Path(
            "../data/training_ready_data/train_and_evaluation_set_{}.json".format(self.question_label)).read_bytes())

    @staticmethod
    def _f1_score(expected_tokens, predicted_tokens, correct_predicted_tokens):
        if correct_predicted_tokens == 0:
            return 0

        recall = correct_predicted_tokens / expected_tokens
        precision = correct_predicted_tokens / predicted_tokens
        f1 = (2 * recall * precision) / (recall + precision)
        return recall, precision, f1
