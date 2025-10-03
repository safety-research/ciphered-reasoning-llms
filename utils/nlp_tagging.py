from functools import lru_cache

from flair.data import Sentence
from flair.models import SequenceTagger
from flair.splitter import SegtokSentenceSplitter

import torch


@lru_cache
def get_sequence_tagger():
    torch.set_num_threads(16)
    return SequenceTagger.load("flair/pos-english-fast").to("cpu")


@lru_cache
def get_sentence_splitter():
    torch.set_num_threads(16)
    return SegtokSentenceSplitter()


def count_pos(s, pos_tags, ignore_sentence_if_found=[]):
    sentence_splitter = get_sentence_splitter()
    sequence_tagger = get_sequence_tagger()

    l_sentences = sentence_splitter.split(s)

    ct = 0
    for sentence in l_sentences:
        do_continue = False
        for val in ignore_sentence_if_found:
            if val in sentence.text:
                do_continue = True
                break
        if do_continue:
            continue

        sequence_tagger.predict(sentence)

        for label in sentence.get_labels():
            if label.value in pos_tags:
                ct += 1

    return ct


def count_num_sentences(s):
    return count_pos(s, ["."])


def count_num_nouns(s, ignore_sentence_if_found):
    return count_pos(
        s,
        ["NN", "NNP", "NNPS", "NNS"],
        ignore_sentence_if_found=ignore_sentence_if_found,
    )


def count_num_verbs(s, ignore_sentence_if_found):
    return count_pos(
        s,
        ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
        ignore_sentence_if_found=ignore_sentence_if_found,
    )
