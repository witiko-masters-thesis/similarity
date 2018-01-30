"""This module contains the language model that maps token lists to vector-space representations."""
from __future__ import division
from heapq import heappush, heappop
import logging
from math import sqrt, floor, ceil
from multiprocessing import cpu_count, Pool
import pickle
import re

from datetime import timedelta
from time import monotonic as time
from gensim import corpora, models
from gensim.matutils import cossim
from numpy import mean, seterr, save, load, diag
from scipy.sparse import lil_matrix, identity, save_npz, load_npz, diags, coo_matrix
# from sparsesvd import sparsesvd

from filenames import UNANNOTATED_DATASET_FNAME, \
    UNANNOTATED_DATASET_DICTIONARY_FNAME, \
    UNANNOTATED_DATASET_DOCUMENT_TERM_MATRIX_FNAME as C_FNAME, \
    UNANNOTATED_DATASET_WEIGHTED_DOCUMENT_TERM_MATRIX_FNAME as W_C_FNAME, \
    UNANNOTATED_DATASET_SOFT_WEIGHTED_DOCUMENT_TERM_MATRIX_FNAME as M_W_C_FNAME, \
    UNANNOTATED_DATASET_TFIDF_FNAME, \
    UNANNOTATED_DATASET_TFIDF_MATRIX_FNAME as W_FNAME, \
    UNANNOTATED_DATASET_W2V_FNAME, \
    UNANNOTATED_DATASET_SOFT_MREL_MATRIX_FNAME as MREL_FNAME, \
    UNANNOTATED_DATASET_SOFT_MLEV_MATRIX_FNAME as MLEV_FNAME, \
    UNANNOTATED_DATASET_SOFT_SVD_MATRIX_UT as SOFT_UT_FNAME, \
    UNANNOTATED_DATASET_SOFT_SVD_MATRIX_S as SOFT_S_FNAME, \
    UNANNOTATED_DATASET_SOFT_SVD_MATRIX_VT as SOFT_VT_FNAME, \
    UNANNOTATED_DATASET_SVD_MATRIX_UT as UT_FNAME, \
    UNANNOTATED_DATASET_SVD_MATRIX_S as S_FNAME, \
    UNANNOTATED_DATASET_SVD_MATRIX_VT as VT_FNAME, \
    EXTERNAL_TERM_SIMILARITY_MODEL_FILENAMES
from preprocessing import documents, SegmentIterator
from workers import levsim

W2V_RANDOM_STATE = 12345
FLOAT_DTYPE = "float32"

LSI_NUM_FEATURES = 300
W2V_NUM_WORKERS = cpu_count()
W2V_NUM_FEATURES = 300
MREL_POWER_FACTOR = 2.0
MLEV_NUM_WORKERS = cpu_count()
MLEV_POOL = Pool(MLEV_NUM_WORKERS)
MLEV_MAX_LENGTH_RATIO = 1.5

LOGGER = logging.getLogger(__name__)

def density(M):
    """Returns the density of a sparse matrix M."""
    return M.getnnz() / (M.shape[0] * M.shape[1])

class LanguageModel(object):
    """A tf-idf language model using the unannotated SemEval 2016/2017 Task 3 dataset."""
    def __init__(self, similarity, technique="hard_topics", soft_matrices=[("mrel", 1.0)],
                 w2v_min_count=5, m_knn=100, m_threshold=0.0, term_similarity="w2v.ql"):
        """
            Sets up a tf-idf language model using the unannotated SemEval 2016/2017 Task 3 dataset.

        Attributes:
            similarity      The similarity model that will be used to compute the similarity
                            between two documents.

            technique       The model that will be employed when computing the similarity of two
                            documents. The following values are admissible:
                              (i) "hard_terms" -- cosine similarity in term space,
                             (ii) "soft_terms" -- soft cosine similarity in term space,
                            (iii) "hard_topics" -- cosine similarity in topic space,
                             (iv) "soft_topics" -- cosine similarity in soft topic space.

            soft_matrices   An iterable of (weight, metric) 2-tuples that specifies a weighted
                            average of similarity matrices that we will be using to model soft terms
                            and topics. The following similarity matrices are available:
                              (i) "mrel" -- mij = max(cossim(vi, vj), m_threshold)**MREL_POWER_FACTOR,
                                  where vi, vj are word2vec vectors corresponding to terms wi, wj.
                             (ii) "mlev" -- mij = MLEV_ALPHA*((1-edit_distance(wi, wj))
                                  /max(|wi|, |wj|))**MLEV_BETA, where |wi|, |wj| are the character
                                  lengths of terms wi and wj.
            
            w2v_min_count   The minimum number of occurences of a term to be included in the word2vec
                            model dictionary.

            m_knn           The number of nearest neighbors of a term that are considered when
                            building the term similarity matrix M. Note that this imposes an upper
                            limit on the number of nonzero elements in any column / row of M.

            m_threshold     The minimum similarity that is recorded inside the sparse term similarity
                            matrix M.
            
            term_similarity The term similarity model and the associated dictionary that should be
                            used when computing the local part of the similarity between two
                            documents (X^TM) with late weighting. The following values are
                            admissible:
                              (i) "w2v.ql" -- the Qatar Living word2vec model.
                             (ii) "w2v.googlenews" -- the Google News word2vec model.
                            (iii) "glove.enwiki_gigaword5" -- the English Wikipedia 2014 +
                                  Gigaword 5 glove model.
                             (iv) "glove.common_crawl" -- the Common Crawl dictionary glove model.
                              (v) "glove.twitter" -- the Twitter glove model.
                             (vi) "fasttext.enwiki" -- the English Wikipedia fasttext model.

        """
        assert technique in ("hard_terms", "soft_terms", "hard_topics", "soft_topics")
        self.technique = technique
        assert isinstance(similarity, Similarity)
        self.similarity = similarity
        assert isinstance(w2v_min_count, int)
        assert isinstance(m_knn, int)
        assert isinstance(m_threshold, float)
        if technique == "soft_terms" or technique == "soft_topics":
            assert soft_matrices
            soft_matrices_config_string = ','.join(["%s_%.10f" % (matrix, weight) \
                                                    for matrix, weight in soft_matrices])
            use_mrel = False
            mrel_weight = 0.0
            use_mlev = False
            mlev_weight = 0.0
            for matrix, weight in soft_matrices:
                assert matrix in ("mrel", "mlev")
                if matrix == "mrel":
                    use_mrel = True
                    mrel_weight = weight
                else:
                    use_mlev = True
                    mlev_weight = weight
            assert use_mrel or use_mlev
        assert term_similarity in ("w2v.ql", "w2v.googlenews", "glove.enwiki_gigaword5",
                                   "glove.common_crawl", "glove.twitter", "fasttext.enwiki")

        try:
            self.dictionary = corpora.Dictionary.load(UNANNOTATED_DATASET_DICTIONARY_FNAME,
                                                      mmap='r')
        except IOError:
            LOGGER.info("Building the dictionary.")
            file_handler = logging.FileHandler("%s.log" % UNANNOTATED_DATASET_DICTIONARY_FNAME,
                                               encoding='utf8')
            logging.getLogger().addHandler(file_handler)
            start_time = time()

            self.dictionary = corpora.Dictionary(document.tokens for document in documents())
            avg_tokens, avg_terms = mean(list(zip(*((len(document.tokens), len(document.terms)) \
                                                    for document in documents()))), axis=1)
            LOGGER.info("Average number of tokens per a document: %f" % avg_tokens)
            LOGGER.info("Average number of terms per a document:  %f" % avg_terms)
            self.dictionary.save(UNANNOTATED_DATASET_DICTIONARY_FNAME)
            self.dictionary = corpora.Dictionary.load(UNANNOTATED_DATASET_DICTIONARY_FNAME,
                                                      mmap='r')

            LOGGER.info("Time elapsed: %s" % timedelta(seconds=time()-start_time))
            logging.getLogger().removeHandler(file_handler)
        m = len(self.dictionary) # number of terms
        self.m = m
        n = self.dictionary.num_docs # number of documents
        self.n = n

        try:
            self.tfidf = models.TfidfModel.load(UNANNOTATED_DATASET_TFIDF_FNAME, mmap='r')
        except IOError:
            LOGGER.info("Building the tf-idf model.")
            file_handler = logging.FileHandler("%s.log" % UNANNOTATED_DATASET_TFIDF_FNAME,
                                               encoding='utf8')
            logging.getLogger().addHandler(file_handler)
            start_time = time()

            corpus_bow = [self.dictionary.doc2bow(document.tokens) for document in documents()]
            self.tfidf = models.TfidfModel(corpus_bow)
            self.tfidf.save(UNANNOTATED_DATASET_TFIDF_FNAME)
            self.tfidf = models.TfidfModel.load(UNANNOTATED_DATASET_TFIDF_FNAME, mmap='r')

            LOGGER.info("Time elapsed: %s" % timedelta(seconds=time()-start_time))
            logging.getLogger().removeHandler(file_handler)

        try:
            self.W = load_npz("%s.npz" % W_FNAME)
        except:
            LOGGER.info("Building the diagonal IDF matrix W.")
            file_handler = logging.FileHandler("%s.log" % W_FNAME, encoding='utf8')
            logging.getLogger().addHandler(file_handler)
            start_time = time()

            W = lil_matrix((m, m), dtype=FLOAT_DTYPE)
            for i in range(m):
                W[i,i] = self.tfidf.idfs[i]
            self.W = W.tocoo()
            save_npz("%s.npz" % W_FNAME, self.W)

            LOGGER.info("Time elapsed: %s" % timedelta(seconds=time()-start_time))
            logging.getLogger().removeHandler(file_handler)
        self.W = self.W.todia()
        del self.tfidf

        if technique == "soft_terms" or technique == "soft_topics":
            self.M = lil_matrix((m, m), dtype=FLOAT_DTYPE)

            if use_mrel:
                if term_similarity == "w2v.ql":
                    w2v_full_fname = "%s-%d" % (UNANNOTATED_DATASET_W2V_FNAME, w2v_min_count)
                    try:
                        self.term_similarity = models.Word2Vec.load(w2v_full_fname, mmap='r').wv
                    except IOError:
                        LOGGER.info("Building the word2vec model.")
                        file_handler = logging.FileHandler("%s.log" % w2v_full_fname, encoding='utf8')
                        logging.getLogger().addHandler(file_handler)
                        start_time = time()
            
                        self.term_similarity = models.Word2Vec(sentences=SegmentIterator(),
                                                               size=W2V_NUM_FEATURES,
                                                               seed=W2V_RANDOM_STATE,
                                                               min_count=w2v_min_count, sg=0,
                                                               workers=W2V_NUM_WORKERS)
                        self.term_similarity.save(w2v_full_fname)
                        self.term_similarity = models.Word2Vec.load(w2v_full_fname, mmap='r').wv
            
                        LOGGER.info("Time elapsed: %s" % timedelta(seconds=time()-start_time))
                        LOGGER.info("Number of terms in the model: %d" % len(self.term_similarity.vocab))
                        logging.getLogger().removeHandler(file_handler)
                elif term_similarity in ("glove.enwiki_gigaword5", "glove.common_crawl",
                                         "glove.twitter", "fasttext.enwiki"):
                    self.term_similarity = models.KeyedVectors.load_word2vec_format( \
                            EXTERNAL_TERM_SIMILARITY_MODEL_FILENAMES[term_similarity], binary=False)
                elif term_similarity == "w2v.googlenews":
                    self.term_similarity = models.KeyedVectors.load_word2vec_format( \
                            EXTERNAL_TERM_SIMILARITY_MODEL_FILENAMES[term_similarity], binary=True)
                m_rel = len(self.term_similarity.vocab) # number of terms in the term similarity model

                Mrel_full_fname = "%s-%s-%d-%d-%f-%f" % (MREL_FNAME, term_similarity, w2v_min_count,
                                                         m_knn, m_threshold, MREL_POWER_FACTOR)
                try:
                    self.Mrel = load_npz("%s.npz" % Mrel_full_fname)
                except FileNotFoundError:
                    LOGGER.info("Building the term similarity matrix Mrel.")
                    file_handler = logging.FileHandler("%s.log" % Mrel_full_fname, encoding='utf8')
                    logging.getLogger().addHandler(file_handler)
                    start_time = time()
        
                    Mrel = identity(m, dtype=FLOAT_DTYPE, format="lil")
                    for k, term_i in enumerate(self.term_similarity.vocab.keys()):
                        if k % 10000 == 0:
                            LOGGER.info("Processing term number %d." % (k+1))
                        i = self.dictionary.doc2bow([term_i])
                        if not i:
                            continue
                        for _, (term_j, similarity) in \
                                zip(range(m_knn),
                                    self.term_similarity.most_similar(positive=[term_i], topn=m_knn)):
                            j = self.dictionary.doc2bow([term_j])
                            if not j:
                                continue
                            if similarity > m_threshold:
                                Mrel[i[0][0],j[0][0]] = similarity**2
                    self.Mrel = Mrel.tocoo()
                    save_npz("%s.npz" % Mrel_full_fname, self.Mrel)
        
                    LOGGER.info("Time elapsed: %s" % timedelta(seconds=time()-start_time))
                    LOGGER.info("Matrix density:\n- %.10f by word2vec,\n- %.10f by kNN," \
                                % (m_rel**2/m**2, ((m_knn+1)*m_rel + 1*(m-m_rel))/m**2) \
                                + "\n- %.10f by thresholding" % density(self.Mrel))
                    logging.getLogger().removeHandler(file_handler)
                del self.term_similarity
                self.M = self.M + mrel_weight * self.Mrel
                del self.Mrel

            if use_mlev:
                Mlev_full_fname = "%s-%d-%f" % (MLEV_FNAME, m_knn, m_threshold)
                try:
                    self.Mlev = load_npz("%s.npz" % Mlev_full_fname)
                except FileNotFoundError:
                    LOGGER.info("Building the term similarity matrix Mlev.")
                    file_handler = logging.FileHandler("%s.log" % Mlev_full_fname, encoding='utf8')
                    logging.getLogger().addHandler(file_handler)
                    start_time = time()
        
                    Mlev = identity(m, dtype=FLOAT_DTYPE, format="lil")
                    min_terms = m
                    avg_terms = []
                    max_terms = 0
                    for k, (i, term_i) in enumerate(self.dictionary.items()):
                        if k % 10 == 0:
                            LOGGER.info("Processing term number %d." % (k+1))
                        terms = [(term_i, term_j, j) for j, term_j \
                                 in self.dictionary.items() \
                                 if i != j and max(len(term_i), len(term_j)) \
                                 / min(len(term_i), len(term_j)) < MLEV_MAX_LENGTH_RATIO]
                        Mlev_chunksize = max(1, ceil(len(terms)/MLEV_NUM_WORKERS))
                        similarities = []
                        for term_num, (similarity, term_j, j) in \
                                enumerate(MLEV_POOL.imap_unordered(levsim, terms, Mlev_chunksize)):
                            heappush(similarities, (-similarity, term_j, j))
                        min_terms = min(min_terms, term_num+1)
                        avg_terms.append(term_num+1)
                        max_terms = max(max_terms, term_num+1)
                        for similarity, term_j, j in (heappop(similarities) for _ \
                                                      in range(min(m_knn, len(similarities)))):
                            similarity = -similarity
                            if similarity > m_threshold:
                                Mlev[i,j] = similarity
                    self.Mlev = Mlev.tocoo()
                    save_npz("%s.npz" % Mlev_full_fname, self.Mlev)
        
                    LOGGER.info("Time elapsed: %s" % timedelta(seconds=time()-start_time))
                    LOGGER.info("Minimum number of terms considered: %d", min_terms)
                    LOGGER.info("Average number of terms considered: %d", mean(avg_terms))
                    LOGGER.info("Maximum number of terms considered: %d", max_terms)
                    LOGGER.info("Matrix density:\n- %.10f by kNN," % (((m_knn+1)*m)/m**2) \
                                + "\n- %.10f by thresholding" % density(self.Mlev))
                    logging.getLogger().removeHandler(file_handler)
                self.M = self.M + mlev_weight * self.Mlev
                del self.Mlev

        if technique == "hard_topics" or technique == "soft_topics":
            try:
                self.C = load_npz("%s.npz" % C_FNAME)
            except FileNotFoundError:
                LOGGER.info("Building the (unweighted) term-document matrix C.")
                file_handler = logging.FileHandler("%s.log" % C_FNAME, encoding='utf8')
                logging.getLogger().addHandler(file_handler)
                start_time = time()
     
                Ct = lil_matrix((n, m), dtype=FLOAT_DTYPE)
                for i, document in enumerate(documents()):
                    if i % 10000 == 0:
                        LOGGER.info("Processing document number %d." % (i+1))
                    for j, ct_ij in self.dictionary.doc2bow(document.tokens):
                        Ct[i,j] = ct_ij
                self.C = Ct.tocoo().transpose()
                del Ct
                save_npz(C_FNAME, self.C)
     
                LOGGER.info("Time elapsed: %s" % timedelta(seconds=time()-start_time))
                LOGGER.info("Matrix density: %f" % density(self.C))
                logging.getLogger().removeHandler(file_handler)
    
            W_C_full_fname = "%s-%d-%d-%f-%f" % (W_C_FNAME, w2v_min_count, m_knn, \
                                                 m_threshold, MREL_POWER_FACTOR)
            try:
                self.W_C = load_npz("%s.npz" % W_C_full_fname)
            except FileNotFoundError:
                LOGGER.info("Building the weighted term-document matrix W*C.")
                file_handler = logging.FileHandler("%s.log" % W_C_full_fname, encoding='utf8')
                logging.getLogger().addHandler(file_handler)
                start_time = time()
    
                W_C = self.W.tocsr().dot(self.C.tocsc())
                self.W_C = W_C.tocoo()
                save_npz("%s.npz" % W_C_full_fname, self.W_C)
    
                LOGGER.info("Time elapsed: %s" % timedelta(seconds=time()-start_time))
                LOGGER.info("Matrix density: %f" % density(self.W_C))
                logging.getLogger().removeHandler(file_handler)
            del self.C
            del self.W

        if technique == "soft_topics":
            M_W_C_full_fname = "%s-%s-%s-%d-%d-%f-%f" % (M_W_C_FNAME, soft_matrices_config_string, \
                                                         term_similarity, \
                                                         w2v_min_count, m_knn, m_threshold, \
                                                         MREL_POWER_FACTOR)
            try:
                self.M_W_C = load_npz("%s.npz" % M_W_C_full_fname)
            except FileNotFoundError:
                LOGGER.info("Building the weighted soft term-document matrix M*W*C.")
                file_handler = logging.FileHandler("%s.log" % Mrel_W_C_full_fname, encoding='utf8')
                logging.getLogger().addHandler(file_handler)
                start_time = time()
    
                M_W_C = self.M.tocsr().dot(self.W_C.tocsc())
                self.M_W_C = M_W_C.tocoo()
                save_npz("%s.npz" % Mrel_W_C_full_fname, self.M_W_C)
    
                LOGGER.info("Time elapsed: %s" % timedelta(seconds=time()-start_time))
                LOGGER.info("Matrix density: %f" % density(self.M_W_C))
                logging.getLogger().removeHandler(file_handler)
            del self.W_C
            del self.M

            soft_Ut_full_fname = "%s-%s-%s-%d-%d-%f-%f" % (SOFT_UT_FNAME, soft_matrices_config_string,\
                                                           term_similarity, w2v_min_count, m_knn, \
                                                           m_threshold, MREL_POWER_FACTOR)
            soft_S_full_fname = "%s-%s-%s-%d-%d-%f-%f" % (SOFT_S_FNAME, soft_matrices_config_string, \
                                                          term_similarity, w2v_min_count, m_knn, \
                                                          m_threshold, MREL_POWER_FACTOR)
            soft_Vt_full_fname = "%s-%s-%s-%d-%d-%f-%f" % (SOFT_VT_FNAME, soft_matrices_config_string,\
                                                           term_similarity, w2v_min_count, m_knn, \
                                                           m_threshold, MREL_POWER_FACTOR)
            try:
                self.UT = load("%s.npy" % soft_Ut_full_fname)
                self.S = load("%s.npy" % soft_S_full_fname)
                self.VT = load("%s.npy" % soft_Vt_full_fname)
            except FileNotFoundError:
                LOGGER.info("Building the SVD of M*W*C.")
                file_handler = logging.FileHandler("%s.log" % soft_Ut_full_fname, encoding='utf8')
                logging.getLogger().addHandler(file_handler)
                start_time = time()
    
                self.UT, self.S, self.VT = sparsesvd(self.M_W_C.tocsc(), LSI_NUM_FEATURES)
                save("%s.npy" % soft_Ut_full_fname, self.UT)
                save("%s.npy" % soft_S_full_fname, self.S)
                save("%s.npy" % soft_Vt_full_fname, self.VT)
    
                LOGGER.info("Time elapsed: %s" % timedelta(seconds=time()-start_time))
                logging.getLogger().removeHandler(file_handler)
            del self.M_W_C

        if technique == "hard_topics":
            try:
                self.UT = load("%s.npy" % UT_FNAME)
                self.S = load("%s.npy" % S_FNAME)
                self.VT = load("%s.npy" % VT_FNAME)
            except FileNotFoundError:
                LOGGER.info("Building the SVD of W*C.")
                file_handler = logging.FileHandler("%s.log" % Ut_full_fname, encoding='utf8')
                logging.getLogger().addHandler(file_handler)
                start_time = time()
    
                self.UT, self.S, self.VT = sparsesvd(self.W_C.tocsc(), LSI_NUM_FEATURES)
                save("%s.npy" % UT_FNAME, self.UT)
                save("%s.npy" % S_FNAME, self.S)
                save("%s.npy" % VT_FNAME, self.VT)
    
                LOGGER.info("Time elapsed: %s" % timedelta(seconds=time()-start_time))
                logging.getLogger().removeHandler(file_handler)
            del self.W_C

        if technique == "hard_topics" or technique == "soft_topics":
            self.Sinv_UT = diag(1/self.S).dot(self.UT)
            del self.UT
            del self.S
            del self.VT

    def sparse2scipy(self, input):
        """Converts a sparse key-value list representation of a document to a sparse scipy array."""
        col = [0] * len(input)
        row, data = zip(*input)
        return coo_matrix((data, (row, col)), shape=(self.m, 1), dtype=FLOAT_DTYPE)

    def compare(self, query, result):
        """Returns similarity between a query and a result document."""
        X = self.sparse2scipy(self.dictionary.doc2bow(query.qsubject.tokens + query.qbody.tokens))
        Y = self.sparse2scipy(self.dictionary.doc2bow(result.qsubject.tokens + result.qbody.tokens))
        if self.technique == "hard_topics" or self.technique == "soft_topics":
            X = self.Sinv_UT * X
            Y = self.Sinv_UT * Y
        return self.similarity.compare(self, X, Y)

class Similarity(object):
    """An interface for an object that represents some measure of similarity between two
       documents."""
    def compare(self, language_model, X, Y):
        """Computes cosine similarity between the query vector X and a result vector Y, where
           language_model is a language model."""

class TopicCosineSimilarity(Similarity):
    """A class that represents the cosine similarity between two documents
       represented by dense topic vectors."""
    def __init__(self):
        """Sets up an object that represents the cosine similarity between two documents."""
        pass

    def compare(self, language_model, X, Y):
        """Computes cosine similarity between the query vector X and a result vector Y, where
           language_model is a language model that provides the term weighting matrix."""
        X_tX = (X.T.dot(X))[0,0]
        Y_tY = (Y.T.dot(Y))[0,0]
        if X_tX == 0.0 or Y_tY == 0.0:
            return 0.0
        X_tY = (X.T.dot(Y))[0,0]
        result = X_tY / (sqrt(X_tX) * sqrt(Y_tY))
        return result

class TermHardCosineSimilarity(Similarity):
    """A class that represents the cosine similarity between two documents
       represented by sparse term vectors."""
    def __init__(self):
        """Sets up an object that represents the cosine similarity between two documents."""
        pass

    def compare(self, language_model, X, Y):
        """Computes cosine similarity between the query vector X and a result vector Y, where
           language_model is a language model that provides the term weighting matrix."""
        WX = language_model.W.tocsr() * X.tocsc()
        WY = language_model.W.tocsr() * Y.tocsc()
        _WX_tWX = (WX.transpose().tocsr() * WX.tocsc())[0,0]
        _WY_tWY = (WY.transpose().tocsr() * WY.tocsc())[0,0]
        if _WX_tWX == 0.0 or _WY_tWY == 0.0:
            return 0.0
        _WX_tWY = (WX.transpose().tocsr() * WY.tocsc())[0,0]
        result = _WX_tWY / (sqrt(_WX_tWX) * sqrt(_WY_tWY))
        return result

class TermSoftCosineSimilarity(Similarity):
    """A class that represents the soft cosine similarity between two documents
       represented by sparse term vectors."""
    def __init__(self, weighting="early", rounding=None, normalization="soft"):
        """Sets up an object that represents the soft cosine similarity between two documents.
        Attributes:
            weighting       Whether a query vector will be weighted before its transpose has been
                            multiplied with the term similarity matrix ("early"), after ("late"),
                            or never (None).

            rounding        Whether the term frequencies in the query vector will be rounded
                            ("round", "ceil", "floor") after the vector's transpose has been
                            multiplied with the term similarity matrix or not (None). The rounding
                            will only be applied with the "late" weighting.

            normalization   Whether the final product will be normalized using the soft cosine
                            norm ("soft"), just the cosine norm ("hard"), or not at all (None).
        """
        assert weighting in ("early", "late", None)
        self.weighting = weighting
        if self.weighting == "early":
            assert rounding is None
            self.rounding = None
        else:
            assert rounding in (None, "round", "ceil", "floor")
            if rounding == "round":
                self.rounding = round
            elif rounding == "ceil":
                self.rounding = ceil
            else:
                self.rounding = floor
        assert normalization in ("soft", "hard", None)
        self.normalization = normalization

    def compare(self, language_model, X, Y):
        """Computes cosine similarity between the query vector X and a result vector Y, where
           language_model is a language model that provides the term weighting and term similarity
           matrices."""
        # Precompute commonly used data.
        if self.weighting is None:
            _WX_tM = (X.transpose().tocsr() * language_model.M.tocsc())
        else:
            WX = language_model.W.tocsr() * X.tocsc()
            WY = language_model.W.tocsr() * Y.tocsc()
            if self.weighting == "early":
                _WX_tM = (WX.transpose().tocsr() * language_model.M.tocsc())
            else:
                XtM = X.transpose().tocsr() * language_model.M.tocsc()
                if self.rounding is not None:
                    XtM = XtM.tocsr()
                    for coord in zip(*XtM.nonzero()):
                        XtM[coord] = self.rounding(XtM[coord])
                W_XtM_t = language_model.W.tocsr() * XtM.transpose().tocsc()

        # Compute the norm.
        if self.normalization == "soft":
            if self.weighting is None or self.weighting == "early":
                if self.weighting is None:
                    _WY_tM = (Y.transpose().tocsr() * language_model.M.tocsc())
                    _WX_tMWX = (_WX_tM.tocsr() * X.tocsc())[0,0]
                    _WY_tMWY = (_WY_tM.tocsr() * Y.tocsc())[0,0]
                elif self.weighting == "early":
                    _WY_tM = (WY.transpose().tocsr() * language_model.M.tocsc())
                    _WX_tMWX = (_WX_tM.tocsr() * WX.tocsc())[0,0]
                    _WY_tMWY = (_WY_tM.tocsr() * WY.tocsc())[0,0]
                if _WX_tMWX == 0.0 or _WY_tMWY == 0.0:
                    return 0.0
                norm = sqrt(_WX_tMWX) * sqrt(_WY_tMWY)
            else:
                YtM = Y.transpose().tocsr() * language_model.M.tocsc()
                W_YtM_t = language_model.W.tocsr() * YtM.transpose().tocsc()
                _W_XtM_t_t_WX = (W_XtM_t.transpose().tocsr() * WX.tocsc())[0,0]
                _W_YtM_t_t_WY = (W_YtM_t.transpose().tocsr() * WY.tocsc())[0,0]
                if _W_XtM_t_t_WX == 0.0 or _W_YtM_t_t_WY == 0.0:
                    return 0.0
                norm = sqrt(_W_XtM_t_t_WX) * sqrt(_W_YtM_t_t_WY)
        elif self.normalization == "hard":
            if self.weighting is None:
                _WX_tWX = (X.transpose().tocsr() * X.tocsc())[0,0]
                _WY_tWY = (Y.transpose().tocsr() * Y.tocsc())[0,0]
            else:
                _WX_tWX = (WX.transpose().tocsr() * WX.tocsc())[0,0]
                _WY_tWY = (WY.transpose().tocsr() * WY.tocsc())[0,0]
            if _WX_tWX == 0.0 or _WY_tWY == 0.0:
                return 0.0
            norm = sqrt(_WX_tWX) * sqrt(_WY_tWY)
        else:
            norm = 1.0

        # Compute the product.
        if self.weighting is None or self.weighting == "early":
            if self.weighting is None:
                _WX_tMWY = (_WX_tM.tocsr() * Y.tocsc())[0,0]
            if self.weighting == "early":
                _WX_tMWY = (_WX_tM.tocsr() * WY.tocsc())[0,0]
            product = _WX_tMWY
        else:
            _W_XtM_t_t_WY = (W_XtM_t.transpose().tocsr() * WY.tocsc())[0,0]
            product = _W_XtM_t_t_WY

        return product / norm
