import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sentence_transformers import InputExample


class SentenceBertDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            train_size=0.9,
            col_sent_1='name_1',
            col_sent_2='name_2',
            col_label='is_duplicate'
    ):
    
        self.train, self.val = train_test_split(df, train_size=train_size)

        labels = torch.FloatTensor(df[col_label].values)
        self.train_examples = []
        for name_1, name_2, label in zip(self.train[col_sent_1], self.train[col_sent_2], labels):
            self.train_examples.append(InputExample(texts=[name_1, name_2], label=label))

    def __len__(self):
        return len(self.train_examples)

    def __getitem__(self, idx):
        return self.train_examples[idx]
