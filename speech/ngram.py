"""N-gram language models measured ON THEIR OWN (before any decoding): perplexity vs order N,
training-set size and smoothing. Day 1 morning.

    lm = NGramLM(order=3, smoothing="interpolated").fit(train_sentences, vocab)
    lm.perplexity(test_sentences)

The lesson: higher N captures longer dependencies but sees each history less often (sparsity), so the
best N depends on the amount of text and on smoothing. The decoding graph also grows with N (lecture).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

BOS, EOS = "<s>", "</s>"


class NGramLM:
    def __init__(self, order: int = 2, smoothing: str = "add-k", k: float = 0.1, lambdas: list | None = None):
        assert order >= 1 and smoothing in ("add-k", "interpolated")
        self.order, self.smoothing, self.k = order, smoothing, k
        # interpolation weights for orders N, N-1, ..., 1 (sum to 1)
        self.lambdas = lambdas or [0.6, 0.3, 0.1, 0.05, 0.05][:order]
        self.lambdas = [w / sum(self.lambdas) for w in self.lambdas]

    def fit(self, sentences: list[list[str]], vocab: list[str]):
        """Count n-grams of every order 1..N. Each sentence is padded with N-1 BOS and one EOS.
        self.counts[n][ngram_tuple] = count, self.context[n][history_tuple] = count (n = 1..N)."""
        self.vocab = list(vocab) + [EOS]

        self.counts = {
            n: Counter() for n in range(1, self.order + 1)
        }
        self.context = {
            n: Counter() for n in range(1, self.order + 1)
        }

        for sentence in sentences:
            # Pad sentence with N-1 BOS tokens and one EOS token
            padded = [BOS] * (self.order - 1) + list(sentence) + [EOS]

            # Count n-grams of every order
            for n in range(1, self.order + 1):
                for i in range(len(padded) - n + 1):
                    ngram = tuple(padded[i:i + n])

                    # BOS is a context token, never a predicted token
                    if ngram[-1] == BOS:
                        continue
                    # Count the n-gram
                    self.counts[n][ngram] += 1

                    # Count its history (context)

                    history = ngram[:-1]
                    self.context[n][history] += 1

        return self

    def prob(self, word: str, history: list[str]) -> float:
        """P(word | last N-1 tokens of history). history may contain BOS tokens."""
        V = len(self.vocab)
        N = self.order
        history = [BOS] * max(0, N - 1 - len(history)) + list(history)
        history = history[-(N - 1):] if N > 1 else []
        # Add-k estimate for a given order
        def add_k(n):
            context_size = n - 1

            h = tuple(history[-context_size:]) if context_size > 0 else ()
            ngram = h + (word,)

            count = self.counts[n][ngram]
            context_count = self.context[n][h]

            return (count + self.k) / (context_count + self.k * V)

        def prob_without_k(n):
            context_size = n - 1

            h = tuple(history[-context_size:]) if context_size > 0 else ()
            ngram = h + (word,)

            count = self.counts[n][ngram]
            context_count = self.context[n][h]

            return (count) / (context_count)


        if self.smoothing == "add-k" and self.k > 0:
            return add_k(N)
                    

        elif self.smoothing == "interpolated":
            # self.lambdas is ordered from N down to 1
            return sum(
                self.lambdas[N - n] * add_k(n)
                for n in range(N, 0, -1)
            )
        
        else:
            return prob_without_k(N)

            

        
    def sentence_logprob(self, sentence: list[str]) -> float:
        hist, lp = [], 0.0
        for w in list(sentence) + [EOS]:
            lp += math.log(self.prob(w, hist))
            hist.append(w)
        return lp

    def perplexity(self, sentences: list[list[str]]) -> float:
        """exp(-(sum of log P) / number of predicted tokens), EOS included."""

        total_log_prob = 0.0
        total_tokens = 0

        for sentence in sentences:
            # Add N-1 BOS tokens to initialize the history
            history = [BOS] * (self.order - 1)

            # Predict every word, then EOS
            for word in list(sentence) + [EOS]:
                p = self.prob(word, history)

                # Avoid log(0)
                if p <= 0:
                    return float("inf")

                total_log_prob += math.log(p)
                total_tokens += 1

                # Update history with the predicted word
                history.append(word)

        if total_tokens == 0:
            return float("inf")

        return math.exp(-total_log_prob / total_tokens)

    
    @property
    def n_entries(self) -> int:
        return sum(len(c) for c in self.counts.values())

    def as_bigram(self, vocab: list[str]) -> dict:
        """Bigram table {u: {v: P(v|u)}} for the word-loop decoder (uses this model's own smoothing)."""
        rows = {}
        for u in [BOS] + list(vocab):
            p = {v: self.prob(v, [u]) for v in vocab}
            z = sum(p.values())
            rows[u] = {v: x / z for v, x in p.items()}
        return rows


def second_order_chain(rng, n_symbols: int = 10, concentration: float = 0.3):
    """A digit 'grammar' where the next digit depends on the previous TWO: a trigram has something to
    learn that a bigram cannot. Returns P[a, b, c] = P(c | a, b), with index n_symbols = start."""
    P = rng.dirichlet(np.full(n_symbols, concentration), size=(n_symbols + 1, n_symbols + 1))
    return P


def sample_second_order(rng, P, n: int, lmin: int = 4, lmax: int = 9) -> list[list[str]]:
    S = P.shape[-1]
    out = []
    for _ in range(n):
        a = b = S  # start symbols
        s = []
        for _ in range(int(rng.integers(lmin, lmax + 1))):
            c = int(rng.choice(S, p=P[a, b]))
            s.append(str(c))
            a, b = b, c
        out.append(s)
    return out
