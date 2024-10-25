import re

class RegexTokenizer:
    """A simple regex-based tokenizer."""
    def __init__(self, vocab: dict[str, int]):
        """
        Args:
            vocab: A dictionary that maps words to their corresponding integer indices."""
        self.vocab = vocab
        self.reverse_vocab: dict[int, str] = {v: k for k, v in vocab.items()}

    def tokenize(self, text: str) -> list[int]:
        """
        Tokenizes a text into a list of integer indices.

        Args:
            text: The text to tokenize.

        Returns:
            A list of integer indices corresponding to the words in the text."""
        res = re.split(r'([,.?_!"()\']|--|\s)', text)
        res = [w.strip() for w in res if w.strip()]
        tokens = [self.vocab[w] for w in res]
        return tokens 

    def detokenize(self, tokens: list[int]) -> str:
        """
        Converts a list of integer indices back into a text.
        
        Args:
            tokens: A list of integer indices.
        """
        text = " ".join([self.reverse_vocab[t] for t in tokens])
        text = re.sub(r' ([,.?_!"()\']|--|\s) ', r'\1', text)
        return text