"""This module provides functions for parsing SemEval 2016/2017 Task 3 datasets."""

from datetime import timedelta
from itertools import chain
import logging
import pickle
import re
from time import monotonic as time
import xml.etree.ElementTree as ElementTree

from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

from filenames import UNANNOTATED_DATASET_FNAME, \
    UNANNOTATED_DATASET_PRECOOKED_FNAME as PRECOOKED_FNAME

CLEANUP_REGEXES = {
    'img': r'<img[^<>]+(>|$)',
    'html': r'<[^<>]+(>|$)',
    'tags': r'\[img_assist[^]]*?\]',
    'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
}

LOGGER = logging.getLogger(__name__)

""" Applies basic preprocessing steps on the text. """
def preprocess(text):
    return [token for token in simple_preprocess(text, min_len=0, max_len=float("inf")) \
            if token not in stopwords.words("english")]

class Document(object):
    """
        A document object that corresponds to <Thread> or <OrgQuestion>
        elements from SemEval 2016/2017 Task 3 datasets.
    """
    def __init__(self, id, segments, qbody, qsubject):
        """
            Sets up a document object that corresponds to <Thread> or
            <OrgQuestion> elements from SemEval 2016/2017 Task 3 datasets.

            id is the unique identifier of a document.

            segments is a list of text segments in the document. Under our
            model, segments correspond to the <OrgQSubject>, <OrgQBody>,
            <RelQSubject>, <RelQBody>, and <RelCText> XML elements.

            qbody is a text segment corresponding to either the
            <OrgQBody>, or the <RelQBody> XML element.

            qsubject is a text segment corresponding to either the
            <OrgQSubject>, or the <RelQsubjec> XML element.

            self.terms contains a set of terms that appear in the document and
            self.tokens contains a list of tokens that appear in the document.

            self.document refers back to self. This allows Document object
            to act as Segment objects in certain situations, such as similarity
            computations.
        """
        assert isinstance(id, str) and isinstance(segments, list) \
               and isinstance(qsubject, Segment) and isinstance(qbody, Segment)
        for segment in segments:
            assert segment.document is None
        self.id = id
        self.segments = segments
        for segment in self.segments:
            segment.document = self
        self.qsubject = qsubject
        self.qbody = qbody
        self.document = self

        # Extract terms and tokens from active segments.
        self.tokens = []
        self.terms = set()
        for segment in segments:
            self.tokens.extend(segment.tokens)
            self.terms.update(segment.terms)

    def __str__(self):
        return ' '.join(self.tokens).__str__()

    def __repr__(self):
        return ' '.join(self.tokens).__repr__()

class Segment(object):
    """
        A document segment object that corresponds to the
        <OrgQSubject>, <OrgQBody>, <RelQSubject>, <RelQBody>, or <RelCText>
        XML element from SemEval 2016/2017 Task 3 datasets.
    """
    def __init__(self, text):
        """
            Sets up a document segment object that corresponds to the
            <OrgQSubject>, <OrgQBody>, <RelQSubject>, <RelQBody>, or <RelCText>
            XML element from SemEval 2016/2017 Task 3 datasets.

            text is the raw unaltered text content of the XML element, which
            is cleaned up and transformed to a list of tokens self.tokens.

            self.terms contains a set of terms that appear in the segment and
            self.tokens contains a list of tokens that appear in the segment.
        """
        assert text is None or isinstance(text, str)
        if text is None:
            self.tokens = []
        else:
            text = re.sub(CLEANUP_REGEXES["img"], "token_img", text)
            text = re.sub(CLEANUP_REGEXES["html"], '', text)
            text = re.sub(CLEANUP_REGEXES["tags"], '', text)
            text = re.sub(CLEANUP_REGEXES["url"], "token_url", text)
            self.tokens = preprocess(text)
        self.terms = set(self.tokens)
        self.document = None

    def __str__(self):
        return ' '.join(self.tokens).__str__()

    def __repr__(self):
        return ' '.join(self.tokens).__repr__()

def segment_orgquestions(dataset_fnames):
    """Segments <OrgQuestion> elements from SemEval 2016/2017 Task 3 datasets."""
    qbody = None
    qsubject = None
    for dataset_fname in dataset_fnames:
        for event, elem in ElementTree.iterparse(dataset_fname):
            if event == "end":
                if elem.tag == "OrgQSubject" or elem.tag == "OrgQBody":
                    segment = Segment(elem.text)
                    if elem.tag == "OrgQSubject":
                        qsubject = segment
                    else:
                        qbody = segment
                elif elem.tag == "OrgQuestion":
                    id = elem.attrib["ORGQ_ID"]
                    assert qbody is not None and qsubject is not None
                    yield Document(id, [qbody, qsubject], qbody, qsubject)
                    qbody = None
                    qsubject = None
            elem.clear()

def segment_threads(dataset_fnames):
    """
        Segments <Thread> elements from SemEval 2016/2017 Task 3 datasets into token lists.
        If full_threads=True, processes entire <Thread>s, otherwise processes
        only the <RelQuestion>s.

        If segment_filtering is not None, a text summarization technique is
        used for the filtering of <Thread> segments.
    """
    segments = []
    relevant = None
    qbody = None
    qsubject = None
    for dataset_fname in dataset_fnames:
        for event, elem in ElementTree.iterparse(dataset_fname):
            if event == "end":
                if elem.tag == "RelQSubject" or elem.tag == "RelQBody" or elem.tag == "RelCText":
                    segment = Segment(elem.text)
                    if elem.tag == "RelQSubject":
                        qsubject = segment
                    if elem.tag == "RelQBody":
                        qbody = segment
                    elif elem.tag == "RelCText":
                        assert segments
                        segments.append(segment)
                elif elem.tag == "RelQuestion":
                    if "RELQ_RELEVANCE2ORGQ" in elem.attrib:
                        relevance_label = elem.attrib["RELQ_RELEVANCE2ORGQ"]
                        relevant = relevance_label == "PerfectMatch" \
                                   or relevance_label == "Relevant"
                    assert qbody is not None and qsubject is not None
                    segments.append(qbody)
                    segments.append(qsubject)
                elif elem.tag == "Thread":
                    id = elem.attrib["THREAD_SEQUENCE"]
                    yield (Document(id, segments, qbody, qsubject),
                           relevant)
                    segments = []
                    qbody = None
                    qsubject = None
            elem.clear()

def retrieve_comment_relevancies(dataset_fnames):
    """
        Extracts the RELC_RELEVANCE2RELQ attributes in <Thread> elements from
        SemEval 2016/2017 Task 3 subtask A datasets.
    """
    relevancies = []
    for dataset_fname in dataset_fnames:
        for event, elem in ElementTree.iterparse(dataset_fname):
            if event == "end":
                if elem.tag == "RelComment":
                    relevance_label = elem.attrib["RELC_RELEVANCE2RELQ"]
                    relevant = relevance_label == "Good" \
                            or relevance_label == "PotentiallyUseful"
                    relevancies.append(relevant)
                elif elem.tag == "Thread":
                    yield relevancies
                    relevancies = []
            elem.clear()

DOCUMENTS=None
def documents():
    """Yields all documents in the unnannotated training Qatar-Living dataset."""
    global DOCUMENTS
    if DOCUMENTS is None:
        try:
            with open(PRECOOKED_FNAME, "br") as file:
                DOCUMENTS = pickle.load(file)
        except FileNotFoundError:
            file_handler = logging.FileHandler("%s.log" % PRECOOKED_FNAME,
                                               encoding='utf8')
            logging.getLogger().addHandler(file_handler)
            start_time = time()
        
            DOCUMENTS = [document for document,_ in \
                         segment_threads([UNANNOTATED_DATASET_FNAME])]
            with open(PRECOOKED_FNAME, "bw") as file:
                dump(DOCUMENTS, file)
        
            LOGGER.info("Time elapsed: %s" % timedelta(seconds=time()-start_time))
            logging.getLogger().removeHandler(file_handler)
    return DOCUMENTS

def segments():
    """Yields all segments in the unnannotated training Qatar-Living dataset."""
    return (segment for document in documents() for segment in document.segments)

class SegmentIterator():
    """Yields all segments in the unnannotated training Qatar-Living dataset."""
    def __iter__(self):
        return segments()
