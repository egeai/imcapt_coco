import nltk
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from typing import List


class Vocabulary(object):
    """Simple vocabulary wrapper"""
    def __init__(self, vocabs: List):
        self.tokenizer = get_tokenizer('basic_english')
        self.vocabs = vocabs

    def yield_tokens(self):
        for vocab in self.vocabs:
            tokens = self.tokenizer(vocab)
            yield tokens

    def build_vocabulary(self) -> Vocab:
        token_generator = self.yield_tokens()
        return build_vocab_from_iterator(token_generator)


class Vocab(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.index = 0

    def __call__(self, token):
        if not token in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[token]

    def __len__(self):
        return len(self.w2i)

    def add_token(self, token):
        if not token in self.w2i:
            self.w2i[token] = self.index
            self.i2w[self.index] = token
            self.index += 1


def build_vocabulary(vocabs: List, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    for i, id in enumerate(vocabs):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i + 1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i + 1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    tokens = [token for token, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocab()
    vocab.add_token('<pad>')
    vocab.add_token('<start>')
    vocab.add_token('<end>')
    vocab.add_token('<unk>')

    # Add the words to the vocabulary.
    for i, token in enumerate(tokens):
        vocab.add_token(token)
    return vocab



# vocab = build_vocab_from_iterator(yield_tokens(file_path), specials=["<unk>"]



