"""This module implements the command-line interface."""

from datetime import timedelta
import logging
from os import path
from sys import argv
from time import monotonic as time

from evaluation import evaluate
from filenames import \
    TEST2016_DATASET_FNAME, TEST2016_DIRNAME, TEST2017_DATASET_FNAME, \
    TEST2017_DIRNAME, TEST_PREDICTIONS_BASE_DIRNAME, \
    TEST2016_PREDICTIONS_DIRNAME, TEST2017_PREDICTIONS_DIRNAME, \
    TEST2016_GOLD_BASE_FNAME, TEST2017_GOLD_BASE_FNAME, \
    DEV_DATASET_FNAME, DEV_GOLD_BASE_FNAME
#   SUBTASK_B_TRAIN2016_DATASET_FNAMES as TRAIN2016_DATASET_FNAMES, \
#   SUBTASK_B_TRAIN2017_DATASET_FNAMES as TRAIN2017_DATASET_FNAMES, \
from language_model import LanguageModel, TopicCosineSimilarity, \
    TermHardCosineSimilarity, TermSoftCosineSimilarity

LOGGER = logging.getLogger(__name__)

def main():
    """This function implements the command-line interface."""
    # Parse input configuration.
    year = argv[2]
    assert year in ("dry_run", "dev", "2016", "2017")
    config = argv[1].split('-', 8)
    technique_string = config[0]
    assert technique_string in ("hard_terms", "soft_terms", "hard_topics", "soft_topics")
    technique = technique_string

    # Set up the document similarity model.
    if technique == "hard_topics" or technique == "soft_topics":
        similarity_model = TopicCosineSimilarity()
    if technique == "soft_topics" or technique == "soft_terms":
        term_similarity_string = config[1]
        assert term_similarity_string in ("w2v.ql", "w2v.googlenews", "glove.enwiki_gigaword5",
                                          "glove.common_crawl", "glove.twitter", "fasttext.enwiki")
        term_similarity = term_similarity_string

        soft_matrices_string = config[2]
        assert soft_matrices_string in ("mrel", "mlev", "mrel_mlev")
        if soft_matrices_string == "mrel":
            soft_matrices = [("mrel", 1.0)]
        elif soft_matrices_string == "mlev":
            soft_matrices = [("mlev", 1.0)]
        else:
            soft_matrices = [("mrel", 0.5), ("mlev", 0.5)]

    if technique == "hard_terms":
        similarity_model = TermHardCosineSimilarity()
        kwargs = {}
    elif technique == "hard_topics":
        kwargs = {}
    elif technique == "soft_terms":
        weighting_string = config[3]
        assert weighting_string in ("early", "late", "none")
        if weighting_string == "none":
            weighting = None
        else:
            weighting = weighting_string

        normalization_string = config[4]
        assert normalization_string in ("soft", "hard", "none")
        if normalization_string == "none":
            normalization = None
        else:
            normalization = normalization_string

        rounding_string = config[5]
        assert rounding_string in ("none", "round", "floor", "ceil")
        if rounding_string == "none":
            rounding = None
        else:
            rounding = rounding_string

        similarity_model = TermSoftCosineSimilarity(weighting=weighting, rounding=rounding, \
                                                    normalization=normalization)

        w2v_min_count=int(config[6])
        m_knn=int(config[7])
        m_threshold=float(config[8])
        kwargs = {"soft_matrices": soft_matrices, "w2v_min_count": w2v_min_count, "m_knn": m_knn, \
                  "m_threshold": m_threshold, "term_similarity": term_similarity }
    elif technique == "soft_topics":
        w2v_min_count=int(config[3])
        m_knn=int(config[4])
        m_threshold=float(config[5])
        kwargs = {"soft_matrices": soft_matrices, "w2v_min_count": w2v_min_count, "m_knn": m_knn, \
                  "m_threshold": m_threshold, "term_similarity": term_similarity }

    if year == "dry_run":
        # Prepare the language model and exit prematurely.
        LanguageModel(similarity=similarity_model, technique=technique, **kwargs)
        return

    # Determine directory and file names.
    if year == "dev":
        test_dirname = TEST2016_DIRNAME
        test_predictions_dirname = TEST2016_PREDICTIONS_DIRNAME
        gold_base_fname = DEV_GOLD_BASE_FNAME
        test_dataset_fname = DEV_DATASET_FNAME
#       train_dataset_fnames = TRAIN2016_DATASET_FNAMES
    elif year == "2016":
        test_dirname = TEST2016_DIRNAME
        test_predictions_dirname = TEST2016_PREDICTIONS_DIRNAME
        gold_base_fname = TEST2016_GOLD_BASE_FNAME
        test_dataset_fname = TEST2016_DATASET_FNAME
#       train_dataset_fnames = TRAIN2016_DATASET_FNAMES + [DEV_DATASET_FNAME]
    elif year == "2017":
        test_dirname = TEST2017_DIRNAME
        test_predictions_dirname = TEST2017_PREDICTIONS_DIRNAME
        gold_base_fname = TEST2017_GOLD_BASE_FNAME
        test_dataset_fname = TEST2017_DATASET_FNAME
#       train_dataset_fnames = TRAIN2017_DATASET_FNAMES + [DEV_DATASET_FNAME]
    output_fname = "%s/subtask_B_%s-%s.txt" % (test_predictions_dirname, argv[1], argv[2])
    base_output_fname = "%s/subtask_B_%s-%s.txt" % (TEST_PREDICTIONS_BASE_DIRNAME, argv[1], argv[2])

    # Perform the evaluation.
    if not path.exists(output_fname):
        LOGGER.info("Producing %s ...", output_fname)
        file_handler = logging.FileHandler("%s.log" % output_fname, encoding='utf8')
        logging.getLogger().addHandler(file_handler)
        start_time = time()

        language_model = LanguageModel(similarity=similarity_model, technique=technique, **kwargs)
        evaluate(language_model, [test_dataset_fname], output_fname)

        LOGGER.info("Time elapsed: %s" % timedelta(seconds=time()-start_time))
        logging.getLogger().removeHandler(file_handler)
    print("%s %s %s" % (test_dirname, gold_base_fname, base_output_fname))

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                        level=logging.INFO)
    main()
