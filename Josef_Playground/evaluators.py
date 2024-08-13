from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from rouge_score import rouge_scorer
from torch import Tensor
from nltk import ngrams
import math
from evaluate import load


class Scorer:
    def __init__(self, n=None):
        self.n = n
        self.n_scores = 0

    def score(self, reference, candidate):
        return Tensor()

    def __str__(self):
        return "Scorer"

    def __call__(self, reference, candidate):
        return self.score(reference, candidate)


class BLEU(Scorer):
    def __init__(self, smoothing_function=False, method=0):
        super().__init__(self)
        self.n_scores = 1
        if smoothing_function:
            if method == 0:
                self.smoothing_function = SmoothingFunction().method0
            elif method == 1:
                self.smoothing_function = SmoothingFunction().method1
            elif method == 2:
                self.smoothing_function = SmoothingFunction().method2
            elif method == 3:
                self.smoothing_function = SmoothingFunction().method3
            elif method == 4:
                self.smoothing_function = SmoothingFunction().method4
            elif method == 5:
                self.smoothing_function = SmoothingFunction().method5
            elif method == 6:
                self.smoothing_function = SmoothingFunction().method6
            elif method == 7:
                self.smoothing_function = SmoothingFunction().method7
            else:
                self.smoothing_function = None

    def score(self, reference, candidate):
        reference = [reference.split()]
        candidate = candidate.split()
        return Tensor([sentence_bleu(reference, candidate, smoothing_function=self.smoothing_function)])

    def __str__(self):
        return "BLEU"


class METEOR(Scorer):
    def __init__(self):
        super().__init__()
        self.n_scores = 1

    def score(self, reference, candidate):
        # Tokenize the reference and candidate sentences
        reference_tokens = nltk.word_tokenize(reference)
        candidate_tokens = nltk.word_tokenize(candidate)

        # Calculate precision, recall, and F1 score
        intersection = len(set(reference_tokens) & set(candidate_tokens))
        precision = intersection / len(candidate_tokens) if len(candidate_tokens) > 0 else 0
        recall = intersection / len(reference_tokens) if len(reference_tokens) > 0 else 0
        harmonic_mean = (10*precision*recall)/(recall+9*precision)
        c = 0
        in_chunk = False
        for ref_token, can_token in zip(reference_tokens, candidate_tokens):
            if ref_token == can_token:
                if not in_chunk:
                    in_chunk = True
                    c += 1
            else:
                in_chunk = False

        # Calculate the number of mapped unigrams (u_m)
        u_m = sum(1 for ref_token, can_token in zip(reference_tokens, candidate_tokens) if ref_token == can_token)

        # Calculate the penalty (p)
        if u_m > 0:
            p = 0.5 * (c / u_m) ** 3
        else:
            p = 0.0

        meteor = harmonic_mean * (1-p)
        return Tensor([meteor])

    def __str__(self):
        return "METEOR"


class Perplexity(Scorer):
    def __init__(self, n=2):
        super().__init__(n=n)
        self.n_scores = 1

    def score(self, reference, candidate):
        reference = reference.split()
        candidate = candidate.split()

        reference_ngrams = list(ngrams(reference, self.n))
        candidate_ngrams = list(ngrams(candidate, self.n))

        reference_freq = nltk.FreqDist(reference_ngrams)

        entropy = 0
        for ngram in candidate_ngrams:
            prob = (reference_freq[ngram] + 1) / (len(reference_ngrams) + len(reference) + 1)
            entropy += -math.log(prob, 2)

        perplexity = 2 ** (entropy / len(candidate_ngrams))
        return Tensor([perplexity])

    def __str__(self):
        return "Perplexity"


class ROUGE(Scorer):
    def __init__(self):
        super().__init__()
        scores = ['rouge1', 'rouge2', 'rougeL']
        self.scorer = rouge_scorer.RougeScorer(scores)

        self.n_scores = 3

    def score(self, reference, candidate):
        scores = self.scorer.score(reference, candidate)
        scores = scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure
        return Tensor(scores)

    def __str__(self):
        return "ROUGE"


class BERT(Scorer):
    def __init__(self):
        super().__init__()
        self.scorer = load('bertscore')

        self.n_scores = 3

    def score(self, reference, candidate):
        scores = self.scorer.compute(predictions=[candidate], references=[reference], lang='en')
        scores = scores['precision'][0], scores['recall'][0], scores['f1'][0]
        return Tensor(scores)

    def __str__(self):
        return "BERT"

