import json
import pandas as pd
from pathlib import Path


class Tokenizer:

    def tokenize(self, tokenizer, question_label, cased_model: bool, training_test_len=None, eval_test_len=None):
        train_contexts, train_questions, train_answers, eval_contexts, eval_questions, eval_answers = self.get_data(
            question_label, cased_model, training_test_len, eval_test_len)

        train_encodings = tokenizer(train_contexts, train_questions, max_length=510, truncation=True, padding=True)
        eval_encodings = tokenizer(eval_contexts, eval_questions, max_length=510, truncation=True, padding=True)

        self.add_token_positions(train_encodings, train_answers)
        self.add_token_positions(eval_encodings, eval_answers)

        return train_encodings, eval_encodings

    @staticmethod
    def add_token_positions(encodings, answers):
        start_positions = []
        end_positions = []

        count = 0
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, int(answers[i]['answer_start'])))
            end_positions.append(encodings.char_to_token(i, int(answers[i]['answer_end'])))

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                count += 1
                start_positions[-1] = 1000

            # if end position is None, the 'char_to_token' function points
            # to the space after the correct token, so add - 1
            if end_positions[-1] is None:
                end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - 1)
                # if end position is still None the answer passage has been truncated
                if end_positions[-1] is None:
                    count += 1
                    end_positions[-1] = 1000

        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    @staticmethod
    def get_data(question_label: str, cased_model: bool, training_test_len=None, eval_test_len=None):
        data = json.loads(
            Path("../data/training_ready_data/train_and_evaluation_set_{}.json".format(question_label)).read_bytes())

        # removing items with bad start/end position for answers
        to_remove = []
        for k, v in data.items():
            for i, item in enumerate(v):
                data[k][i][2] = data[k][i][2].strip()
                # case-sensitive model changes
                if not cased_model:
                    data[k][i][0] = data[k][i][0].strip().lower()
                    data[k][i][1] = data[k][i][1].strip().lower()
                    data[k][i][2] = data[k][i][2].lower()
                if cased_model:
                    data[k][i][1] = data[k][i][1].strip().capitalize()

                # if start or end index in None remove it
                if pd.isna(item[3]) or pd.isna(item[4]):
                    to_remove.append((k, i))

        to_remove_training = [x[1] for x in to_remove if x[0] == "training_set"]
        to_remove_eval = [x[1] for x in to_remove if x[0] == "evaluation_set"]

        data["training_set"] = [item for i, item in enumerate(data["training_set"]) if i not in to_remove_training]
        data["evaluation_set"] = [item for i, item in enumerate(data["evaluation_set"]) if i not in to_remove_eval]

        training_data, evaluation_data = data["training_set"], data["evaluation_set"]
        train_contexts, train_questions, train_answers = [x[0] for x in training_data], [x[1] for x in training_data], [
            x[2:] for x in training_data]
        eval_contexts, eval_questions, eval_answers = [x[0] for x in evaluation_data], [x[1] for x in
                                                                                        evaluation_data], [
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


