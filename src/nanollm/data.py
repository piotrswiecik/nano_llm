import torch
from torch.utils.data import Dataset, DataLoader

from nanollm.token import Tokenizer


class GPTDataset(Dataset):
    """Text dataset for GPT pretraining.

    Contains sliding windows (i+1 slide with stride/step) of tokens from a text.
    For example, given a text "hello world", max_length=5, stride=2:
    inputs = [[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]]
    targets = [[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]]
    """

    def __init__(self, text: str, tokenizer: Tokenizer, max_length: int, stride: int):
        """

        Args:
            text (str): The text to create the dataset from.
            tokenizer (Tokenizer): The tokenizer to use. Must support the `tokenize` method.
            max_length (int): The maximum length of the sliding windows.
            stride (int): The stride/step of the sliding windows.
        """
        tokens = tokenizer.tokenize(text)
        self._inputs = []
        self._targets = []

        for i in range(0, len(tokens) - max_length, stride):
            input_window = tokens[i : i + max_length]
            target_window = tokens[i + 1 : i + max_length + 1]
            self._inputs.append(torch.tensor(input_window))
            self._targets.append(torch.tensor(target_window))

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, index):
        return self._inputs[index], self._targets[index]
