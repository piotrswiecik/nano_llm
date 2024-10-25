import pytest
from nanollm.token import RegexTokenizer


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
    