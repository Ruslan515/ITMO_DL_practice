import torch
import numpy as np
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm

class BertTrain():
    def __init__(
            self,
            epochs,
            train_dataloader,
            val_dataloader,
            model,
            optimizer,
            scheduler,
            device='cuda:0'
    ):

        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.training_stats = []

    def train(self):

        train_labels, val_labels = [], []
        train_logits, val_logits = [], []

        for epoch in tqdm(range(self.epochs)):
            total_train_loss = 0
            self.model.train()

            for input_ids, input_mask, labels in self.train_dataloader:
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                labels = labels.to(self.device)

                self.model.zero_grad()

                outputs = self.model(input_ids,
                                     token_type_ids=None,
                                     attention_mask=input_mask,
                                     labels=labels)

                loss, logits = outputs['loss'], outputs['logits']
                total_train_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

                logits_for_metrics = logits.detach().cpu().numpy()
                logits_for_metrics = np.argmax(logits_for_metrics, axis=1).flatten()

                train_logits.extend(list(logits_for_metrics))
                train_labels.extend(list(labels.cpu().numpy().flatten()))

            avg_train_loss = total_train_loss / len(self.train_dataloader)

            print("")
            print("Average training loss: {0:.2f}".format(avg_train_loss))

            print("")
            print("Running Validation...")

            self.model.eval()

            total_eval_loss = 0

            for input_ids, input_mask, labels in self.val_dataloader:
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids,
                                         token_type_ids=None,
                                         attention_mask=input_mask,
                                         labels=labels)

                    loss, logits = outputs['loss'], outputs['logits']

                total_eval_loss += loss.item()

                logits_for_metrics = logits.detach().cpu().numpy()
                logits_for_metrics = np.argmax(logits_for_metrics, axis=1).flatten()

                val_logits.extend(list(logits_for_metrics))
                val_labels.extend(list(labels.cpu().numpy().flatten()))

            train_f1 = f1_score(train_logits, train_labels, average='macro')
            val_f1 = f1_score(val_logits, val_labels, average='macro')
            print("  train f1_score: {0:.2f}".format(train_f1))
            print("  val f1_score: {0:.2f}".format(val_f1))

            avg_val_loss = total_eval_loss / len(self.val_dataloader)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))

            self.training_stats.append(
                {
                    'epoch': epoch + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Train f1_macro': train_f1,
                    'Valid f1_macro': val_f1
                }
            )

    def __call__(self):
        self.train()
        return self.training_stats
