import nltk
from collections import Counter
from torchtext.data.utils import get_tokenizer
from pycocotools.coco import COCO
from omegaconf import DictConfig
from prefect import flow, task
from dataclasses import dataclass, field

nltk.download('punkt')


@dataclass
class VocabWrapper(object):
    """Simple vocabulary wrapper."""

    w2i = {}
    i2w = {}
    index = 0

    def __call__(self, token):
        if token not in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[token]

    def __len__(self):
        return len(self.w2i)

    def add_token(self, token):
        if token not in self.w2i:
            self.w2i[token] = self.index
            self.i2w[self.index] = token
            self.index += 1


@dataclass
class Vocabulary(object):
    """Simple vocabulary builder
    We build the vocabulary – that is, a dictionary
    that can convert actual textual tokens (such as words) into
    numeric tokens.
    """

    json_path: str
    conf: DictConfig
    coco: COCO = field(init=False)
    counter: Counter = field(init=False)

    def __post_init__(self):
        self.coco = COCO(self.json_path)
        self.counter = Counter()
        self.ids = self.coco.anns.keys()

    def __build_tokens(self):
        for i, id_ in enumerate(self.ids):
            caption = str(self.coco.anns[id_]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            self.counter.update(tokens)
            if (i + 1) % 1000 == 0:
                print("[{}/{}] Tokenized the captions.".format(i + 1, len(self.ids)))

    def __build_vocabs(self):
        """
        First, inside the vocabulary builder function,
        JSON text annotations are loaded, and individual
        words in the annotation/caption are tokenized or
        converted into numbers and stored in a counter.
        :return: vocabulary
        """
        # If the word frequency is less than 'threshold', then the word is discarded.
        tokens = [token for token, cnt in self.counter.items() if cnt >= self.conf.params.vocab_threshold]
        # Create a vocab wrapper and add some special tokens.
        vocab = VocabWrapper()
        vocab.add_token('<pad>')
        vocab.add_token('<start>')
        vocab.add_token('<end>')
        vocab.add_token('<unk>')
        # Add the words to the vocabulary.
        for i, token in enumerate(tokens):
            vocab.add_token(token)
        return vocab

    def build_vocab_flow(self):
        self.__build_tokens()
        return self.__build_vocabs()
