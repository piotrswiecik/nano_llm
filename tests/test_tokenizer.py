import pytest
from nanollm.token import RegexTokenizer, BPETokenizer


def test_regex_tokenizer_should_tokenize_text():
    vocab = {"a": 1, "b": 2, "c": 3}
    tokenizer = RegexTokenizer(vocab)
    text = "a b c"
    tokens = tokenizer.tokenize(text)
    assert tokens == [1, 2, 3]


def test_regex_tokenizer_should_raise_when_word_not_in_vocab():
    vocab = {"a": 1, "b": 2}
    tokenizer = RegexTokenizer(vocab)
    text = "a b c"
    with pytest.raises(KeyError):
        tokenizer.tokenize(text)


def test_bpe_should_tokenize_text():
    tokenizer = BPETokenizer()
    text = "a b c"
    tokens = tokenizer.tokenize(text)
    assert all(isinstance(t, int) for t in tokens)


def test_bpe_should_detokenize_tokens():
    tokenizer = BPETokenizer()
    tokens = [1, 2, 3]
    text = tokenizer.detokenize(tokens)
    assert all(isinstance(t, str) for t in text.split())


def test_bpe_should_handle_eot():
    tokenizer = BPETokenizer()
    eot = "<|endoftext|>"
    tokens = tokenizer.tokenize(eot)
    assert len(tokens) == 1
    assert tokens[0] == 50256


def test_bpe_should_handle_nonsense_word():
    tokenizer = BPETokenizer()
    text = "asdasd123"
    tokens = tokenizer.tokenize(text)
    assert len(tokens) != 0
