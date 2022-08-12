from training.train import TrainAndEvaluation


def main():
    if __name__ == '__main__':
        # https://huggingface.co/distilbert-base-cased-distilled-squad
        # https://huggingface.co/deepset/roberta-base-squad2
        # https://huggingface.co/deepset/minilm-uncased-squad2

        for model, c in [("deepset/minilm-uncased-squad2", False), ("distilbert-base-cased-distilled-squad", True),
                         ("deepset/roberta-base-squad2", True)]:
            for lr in [0.0001, 0.0005]:
                for label in ["", "what", "which", "how"]:
                    # user the training_test_len and eval_test_len to test with smaller number or entries
                    tr = TrainAndEvaluation(model_name=model, question_label=label,
                                            cased_model=c,
                                            learning_rate=lr, epochs=1, training_test_len=40, eval_test_len=20)

                    tr.train()
                    tr.evaluate()


if __name__ == '__main__':
    main()
