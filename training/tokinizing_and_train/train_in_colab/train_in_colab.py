import datetime
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, IntervalStrategy, Trainer
from transformers import default_data_collator
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def load_data(question_label: str, cased_model: bool, training_test_len=None, eval_test_len=None):
    data = json.loads(
        Path("/content/drive/MyDrive/train_in_colab/data_set_split_{}.json".format(question_label)).read_bytes())
    # removing items with bad start/end position for answers
    to_remove = []
    for k, v in data.items():
        for i, item in enumerate(v):
            data[k][i][2] = data[k][i][2].strip()

            if not cased_model:
                data[k][i][0] = data[k][i][0].strip().lower()
                data[k][i][1] = data[k][i][1].strip().lower()
                data[k][i][2] = data[k][i][2].lower()
            if cased_model:
                data[k][i][1] = data[k][i][1].strip().capitalize()
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

    x = training_test_len if training_test_len else len(train_contexts)
    u = eval_test_len if eval_test_len else len(train_contexts)
    return train_contexts[:x], train_questions[:x], train_answers[:x], eval_contexts[:u], eval_questions[
                                                                                          :u], eval_answers[:u]


def add_token_positions(encodings, answers):
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
            start_positions[-1] = 1000

        # if end position is None, the 'char_to_token' function points to the space after the correct token, so add - 1
        if end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - 1)
            # if end position is still None the answer passage has been truncated
            if end_positions[-1] is None:
                count += 1
                end_positions[-1] = 1000

    # todo: what is this doing ?
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


def tokenize(tokenizer, question_label, cased_model: bool, training_test_len=None, eval_test_len=None):
    train_contexts, train_questions, train_answers, eval_contexts, eval_questions, eval_answers = load_data(
        question_label, cased_model, training_test_len, eval_test_len)
    tokenizer = tokenizer

    # todo: what should be done about truncation
    train_encodings = tokenizer(train_contexts, train_questions, max_length=510, truncation=True, padding=True)
    val_encodings = tokenizer(eval_contexts, eval_questions, max_length=510, truncation=True, padding=True)

    add_token_positions(train_encodings, train_answers)
    add_token_positions(val_encodings, eval_answers)

    return train_encodings, val_encodings


class TrainAndEvaluation:

    def __init__(self, model_name: str, question_label: str, cased_model: bool, learning_rate: float, epochs: int,
                 training_test_len=None, eval_test_len=None):

        self.result_model_path = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                   else 'cpu')

        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_encodings, val_encodings = tokenize(self.tokenizer, question_label, cased_model, training_test_len,
                                                  eval_test_len)

        self.train_set = MyDataset(train_encodings)
        self.val_set = MyDataset(val_encodings)

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

    def evaluate(self):
        path = self.result_model_path
        model = AutoModelForQuestionAnswering.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)

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
            text = item[0]
            question = item[1]
            target_answer = item[2].strip().lower()
            total_tokens_expected += len(target_answer.split(" "))

            inputs = tokenizer(question, text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax()

            predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
            predict_answer = tokenizer.decode(predict_answer_tokens)
            predict_answer = predict_answer.strip().lower()

            if not predict_answer:
                no_answer += 1

            containment_match = target_answer in predict_answer
            if containment_match:
                containment += 1

            cleaned_predict_answer = predict_answer

            if cleaned_predict_answer:
                if cleaned_predict_answer[-1] in [".", ",", ";", ":", "-", ")", "(", "_", "+"]:
                    cleaned_predict_answer = cleaned_predict_answer[:-1].strip()
                    total_tokens_predicted += len(cleaned_predict_answer.split(" "))

                    if containment_match:
                        total_correct_tokens_predicted_containment += len(cleaned_predict_answer.split(" "))
                else:
                    if containment_match:
                        total_correct_tokens_predicted_containment += len(predict_answer.split(" "))

            exact_match = cleaned_predict_answer == target_answer
            if exact_match:
                success += 1
                total_correct_tokens_predicted_exact += len(cleaned_predict_answer.split(" "))

            results.append((item[0], item[1], item[2], predict_answer, exact_match, containment_match))

        f1_score_exact = self._f1_score(total_tokens_expected, total_tokens_predicted,
                                        total_correct_tokens_predicted_exact)

        f1_score_containment = self._f1_score(total_tokens_expected, total_tokens_predicted,
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
            "f1_score_exact": f1_score_exact,
            "f1_score_containment": f1_score_containment
        }
        with open(metric_path + "metrics.json", "w") as file:
            json.dump(res_dic, file)

    def _load_data_set(self):
        return json.loads(Path(
            "/content/drive/MyDrive/train_in_colab/data_set_split_{}.json".format(self.question_label)).read_bytes())

    @staticmethod
    def _f1_score(expected_tokens, predicted_tokens, correct_predicted_tokens):
        if correct_predicted_tokens == 0:
            return 0

        recall = correct_predicted_tokens / expected_tokens
        precision = correct_predicted_tokens / predicted_tokens
        f1 = (2 * recall * precision) / (recall + precision)
        return recall, precision, f1


if __name__ == '__main__':
    for label in ["which", "how"]:
        for lr in [0.0001, 0.00005]:
            tr = TrainAndEvaluation(model_name="distilbert-base-cased-distilled-squad", question_label=label,
                                    cased_model=True,
                                    learning_rate=lr, epochs=4)
            tr.train()



