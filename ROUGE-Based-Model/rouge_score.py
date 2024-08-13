import collections
import numpy as np
import pandas as pd
from nltk.util import ngrams
from nltk import word_tokenize, sent_tokenize

# NOTE: Implementation copied from [rouge_score] slightly altered
# NOTE: Re-Implemented for Kaggle (library not available)

Score = collections.namedtuple('Score', ['precision', 'recall', 'fmeasure'])


def _tokenize(text):
    tokens = word_tokenize(text.lower())
    return tokens


def _get_sentences(text):
    sentences = sent_tokenize(text)
    sentences = [x for x in sentences if len(x)]
    return sentences


def _lcs_table(ref, can):
    rows = len(ref)
    cols = len(can)
    lcs_table = np.zeros((rows + 1, cols + 1))
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if ref[i - 1] == can[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
    return lcs_table


def _backtrack_norec(t, ref, can):
    i = len(ref)
    j = len(can)
    lcs = []
    while i > 0 and j > 0:
        if ref[i - 1] == can[j - 1]:
            lcs.insert(0, i - 1)
            i -= 1
            j -= 1
        elif t[i][j - 1] > t[i - 1][j]:
            j -= 1
        else:
            i -= 1
    return lcs


def _lcs_ind(ref, can):
    """Returns one of the longest lcs."""
    t = _lcs_table(ref, can)
    return _backtrack_norec(t, ref, can)


def _find_union(lcs_list):
    """Finds union LCS given a list of LCS."""
    return sorted(list(set().union(*lcs_list)))


def _union_lcs(ref, c_list):
    lcs_list = [_lcs_ind(ref, c) for c in c_list]
    return [ref[i] for i in _find_union(lcs_list)]


def rouge_n(target, prediction, n):
    target_tokens = _tokenize(target)
    prediction_tokens = _tokenize(prediction)

    target_ngrams = collections.Counter(ngrams(target_tokens, n))
    prediction_ngrams = collections.Counter(ngrams(prediction_tokens, n))

    intersection_ngrams_count = 0
    for ngram in target_ngrams:
        intersection_ngrams_count += min(target_ngrams[ngram],
                                         prediction_ngrams[ngram])
    target_ngrams_count = sum(target_ngrams.values())
    prediction_ngrams_count = sum(prediction_ngrams.values())

    precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
    recall = intersection_ngrams_count / max(target_ngrams_count, 1)

    if precision + recall > 0:
        fmeasure = 2 * precision * recall / (precision + recall)
    else:
        fmeasure = 0.0

    return pd.Series({'precision': precision, 'recall': recall, 'fmeasure': fmeasure})


def rouge_l(target, prediction):
    target_tokens = _tokenize(target)
    prediction_tokens = _tokenize(prediction)

    if not target_tokens or not prediction_tokens:
        return pd.Series({'precision': 0, 'recall': 0, 'fmeasure': 0})

    lcs_table = _lcs_table(target_tokens, prediction_tokens)

    lcs_length = lcs_table[-1][-1]

    precision = lcs_length / len(prediction_tokens)
    recall = lcs_length / len(target_tokens)
    if precision + recall > 0:
        fmeasure = 2 * precision * recall / (precision + recall)
    else:
        fmeasure = 0.0

    return pd.Series({'precision': precision, 'recall': recall, 'fmeasure': fmeasure})


def rouge_lsum(target, prediction):
    target_tokens_list = [
        _tokenize(s) for s in _get_sentences(target)]
    prediction_tokens_list = [
        _tokenize(s) for s in _get_sentences(prediction)]

    if not target_tokens_list or not prediction_tokens_list:
        return pd.Series({'precision': 0, 'recall': 0, 'fmeasure': 0})

    m = sum(map(len, target_tokens_list))
    n = sum(map(len, prediction_tokens_list))
    if not n or not m:
        return pd.Series({'precision': 0, 'recall': 0, 'fmeasure': 0})

    # get token counts to prevent double counting
    token_cnts_r = collections.Counter()
    token_cnts_c = collections.Counter()
    for s in target_tokens_list:
        # s is a list of tokens
        token_cnts_r.update(s)
    for s in prediction_tokens_list:
        token_cnts_c.update(s)

    hits = 0
    for r in target_tokens_list:
        lcs = _union_lcs(r, prediction_tokens_list)
        # Prevent double-counting:
        # The paper describes just computing hits += len(_union_lcs()),
        # but the implementation prevents double counting
        for t in lcs:
            if token_cnts_c[t] > 0 and token_cnts_r[t] > 0:
                hits += 1
                token_cnts_c[t] -= 1
                token_cnts_r[t] -= 1

    recall = hits / m
    precision = hits / n
    if precision + recall > 0:
        fmeasure = 2 * precision * recall / (precision + recall)
    else:
        fmeasure = 0.0
    return pd.Series({'precision': precision, 'recall': recall, 'fmeasure': fmeasure})


if __name__ == '__main__':
    print(rouge_n("The quick brown jumps over the lazy dog dog", 'A brown fox jumped over the dog', 1))
