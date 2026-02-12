from typing import Literal

from fugashi import Tagger

TokenizationStrategy = Literal["baseline", "character", "morphology"]


TOKENIZATION_STRATEGIES: list[TokenizationStrategy] = [
    "baseline",
    "character",
    "morphology",
]


class Tokenizer:
    def __init__(self):
        self.tagger = Tagger("-Owakati")

    def tokenize(self, string: str, strategy: TokenizationStrategy):
        if strategy == "baseline":
            return string
        elif strategy == "character":
            return self.de_tokenize_character(string)
        elif strategy == "morphology":
            return self.de_tokenize_morphology(string)

    def de_tokenize_character(self, string: str):
        return " ".join(list(string))

    def de_tokenize_morphology(self, string: str):
        return self.tagger.parse(string).strip()


if __name__ == "__main__":
    tokenizer = Tokenizer()
    print(tokenizer.de_tokenize_morphology("これはテストです。"))
