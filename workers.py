"""This module contains the multiprocessing worker functions."""
import Levenshtein

MLEV_ALPHA = 1.8
MLEV_BETA = 5.0

def levsim(args):
    """Returns the Levenshtein similarity between two terms."""
    term_i, term_j, j = args
    return (MLEV_ALPHA * (1 - Levenshtein.distance(term_i, term_j) \
            / max(len(term_i), len(term_j)))**MLEV_BETA, term_j, j)
