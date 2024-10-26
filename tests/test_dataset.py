import logging

import torch

from nanollm.data import GPTDataset
from nanollm.token import BPETokenizer


def test_gpt_dataset():
    tokenizer = BPETokenizer()
    text = "hello world this is a test text for the dataset creation and testing"
    dataset = GPTDataset(text, tokenizer, max_length=5, stride=2)

    assert len(dataset) == 4
    assert isinstance(dataset[0][0], torch.Tensor)
    assert isinstance(dataset[0][1], torch.Tensor)
    assert all(item[0].shape[0] == 5 for item in dataset)

    items = list(dataset)  # should not raise any errors
    for item in items:
        assert item[0].shape == item[1].shape
        assert item[0][-1] == item[1][-2]  # sliding window works correctly
