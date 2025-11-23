import re
from collections import defaultdict


class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.itos = {
            0: self.pad_token,
            1: self.sos_token,
            2: self.eos_token,
            3: self.unk_token,
        }

        self.stoi = {token: idx for idx, token in self.itos.items()}

        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def _tokenize(text):
        """
        Basic tokenizer:
        - lowercase text
        - keep only alphanumeric characters
        - split on whitespace
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return text.strip().split()

    def build_vocabulary(self, sentences):
        """
        Build the vocabulary from a list of caption strings.
        Words must appear at least `freq_threshold` times to be added.
        """
        frequencies = defaultdict(int)
        idx = 4

        for sentence in sentences:
            for word in self._tokenize(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def numericalize(self, text):
        tokens = self._tokenize(text)
        return [self.stoi.get(token, self.stoi[self.unk_token]) for token in tokens]
