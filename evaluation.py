"""This module contains high-level training and evaluation functions."""

import logging

from preprocessing import segment_threads, segment_orgquestions

LOGGER = logging.getLogger(__name__)

def evaluate(language_model, dataset_fnames, output_fname):
    """Produces an output file that contains the ranking of document pairs."""
    with open(output_fname, "wt") as output_file:
        for orgquestion, (thread, _) in zip(segment_orgquestions(dataset_fnames),
                                            segment_threads(dataset_fnames)):
            test_score = language_model.compare(orgquestion, thread)
            output_file.write("%s\t%s\t0\t%s\ttrue\n" % (orgquestion.id, thread.id, repr(test_score)))
