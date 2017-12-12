.PHONY: all
all: QL-unannotated-data-subtaskA.xml v3.2/dev/SemEval2016-Task3-CQA-QL-dev.xml \
	SemEval2016_task3_test/English/SemEval2016-Task3-CQA-QL-test.xml \
	SemEval2016_task3_submissions_and_scores/_scorer/ev.py \
	SemEval2017_task3_test_input_ABCD/English-ABC/SemEval2017-task3-English-test-input.xml \
	SemEval2017_task3_submissions_and_scores/_scorer/ev.py \
	wiki.en.vec glove.6B.300d.w2v glove.840B.300d.w2v glove.twitter.27B.200d.w2v GoogleNews-vectors-negative300.bin.gz

# SemEval Task 3 datasets
## See <http://alt.qcri.org/semeval2017/task3/index.php?id=data-and-tools>
QL-unannotated-data-subtaskA.xml.zip semeval2016-task3-cqa-ql-traindev-v3.2.zip semeval2016_task3_submissions_and_scores.zip semeval2016_task3_test.zip:
	wget http://alt.qcri.org/semeval2016/task3/data/uploads/$@

semeval2017_task3_test_input_abcd.zip semeval2017_task3_submissions_and_scores.zip:
	wget http://alt.qcri.org/semeval2017/task3/data/uploads/$@
	
QL-unannotated-data-subtaskA.xml: QL-unannotated-data-subtaskA.xml.zip
	unzip -o $<
	test -e $@ && touch $@

v3.2/dev/SemEval2016-Task3-CQA-QL-dev.xml: semeval2016-task3-cqa-ql-traindev-v3.2.zip
	unzip -o $<
	test -e $@ && touch $@

SemEval2016_task3_test/English/SemEval2016-Task3-CQA-QL-test.xml: semeval2016_task3_test.zip
	unzip -o $<
	test -e $@ && touch $@

SemEval2016_task3_submissions_and_scores/_scorer/ev.py: semeval2016_task3_submissions_and_scores.zip
	unzip -o $<
	mkdir SemEval2016_task3_submissions_and_scores/RaRe
	test -e $@ && touch $@

SemEval2017_task3_test_input_ABCD/English-ABC/SemEval2017-task3-English-test-input.xml: semeval2017_task3_test_input_abcd.zip
	unzip -o $<
	test -e $@ && touch $@

SemEval2017_task3_submissions_and_scores/_scorer/ev.py: semeval2017_task3_submissions_and_scores.zip
	unzip -o $<
	mkdir SemEval2017_task3_submissions_and_scores/RaRe
	test -e $@ && touch $@

# Pre-trained FastText word vectors
## See <https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md>
wiki.en.vec: wiki.en.zip
	unzip -o $<
	test -e $@ && touch $@

wiki.en.zip:
	wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip

# Pre-trained Glove word vectors
## See <https://nlp.stanford.edu/projects/glove/> 

%.w2v: %.txt
	python -m gensim.scripts.glove2word2vec --input $< --output $@

### Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download)
glove.6B.300d.txt: glove.6B.zip
	unzip -o $<
	test -e $@ && touch $@

glove.6B.zip:
	wget http://nlp.stanford.edu/data/glove.6B.zip

### Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)
glove.840B.300d.txt: glove.840B.300d.zip
	unzip -o $<
	test -e $@ && touch $@

glove.840B.300d.zip:
	wget http://nlp.stanford.edu/data/glove.840B.300d.zip

### Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download)
glove.twitter.27B.200d.txt: glove.twitter.27B.zip
	unzip -o $<
	test -e $@ && touch $@

glove.twitter.27B.zip:
	wget http://nlp.stanford.edu/data/glove.twitter.27B.zip

# Pre-trained Word2Vec word vectors
## See <https://code.google.com/archive/p/word2vec/>
GoogleNews-vectors-negative300.bin.gz:
	wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
