import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering

from training.tokinizing_and_train.DataSet import MyDataset
from training.tokinizing_and_train.tokinizing import tokenize

train_encodings, val_encodings = tokenize()
train_dataset = MyDataset(train_encodings)
val_dataset = MyDataset(val_encodings)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad").to(device)

optim = AdamW(model.parameters(), lr=5e-5)
# optim = AdamW(model.parameters(), lr=3e-5)
# optim = AdamW(model.parameters(), lr=2e-5)

epochs = 2


def train():
    whole_train_eval_time = time.time()

    train_losses = []
    val_losses = []

    print_every = 1000

    for epoch in range(epochs):
        epoch_time = time.time()

        # Set model in train mode
        model.train()

        loss_of_epoch = 0

        print("############Train############")

        for batch_idx, batch in tqdm(enumerate(train_loader)):

            optim.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                            end_positions=end_positions)
            loss = outputs[0]
            # do a backwards pass
            loss.backward()
            # update the weights
            optim.step()
            # Find the total loss
            loss_of_epoch += loss.item()

            if (batch_idx + 1) % print_every == 0:
                print("Batch {:} / {:}".format(batch_idx + 1, len(train_loader)), "\nLoss:", round(loss.item(), 1),
                      "\n")

        loss_of_epoch /= len(train_loader)
        train_losses.append(loss_of_epoch)

        ##########Evaluation##################

        # Set model in evaluation mode
        model.eval()

        print("############Evaluate############")

        loss_of_epoch = 0

        for batch_idx, batch in enumerate(val_loader):

            with torch.no_grad():

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                                end_positions=end_positions)
                loss = outputs[0]
                # Find the total loss
                loss_of_epoch += loss.item()

            if (batch_idx + 1) % print_every == 0:
                print("Batch {:} / {:}".format(batch_idx + 1, len(val_loader)), "\nLoss:", round(loss.item(), 1), "\n")

        loss_of_epoch /= len(val_loader)
        val_losses.append(loss_of_epoch)

        # Print each epoch's time and train/val loss
        print("\n-------Epoch ", epoch + 1,
              "-------"
              "\nTraining Loss:", train_losses[-1],
              "\nValidation Loss:", val_losses[-1],
              "\nTime: ", (time.time() - epoch_time),
              "\n-----------------------",
              "\n\n")

    print("Total training and evaluation time: ", (time.time() - whole_train_eval_time))


if __name__ == '__main__':
    # train()
    import cProfile

    cProfile.run('train()', sort="tottime")