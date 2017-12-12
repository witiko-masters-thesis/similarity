#!/bin/bash
set -e
export LC_ALL=C
export PARALLEL_SHELL=/bin/bash
WORKERS_NUM=$(nproc)
CONFIGS=(hard_terms
         soft_terms-w2v.ql-mrel-early-soft-none-{{5000,500,50,5,0}-100-0.0,5-{1,10,1000,10000}-0.0,5-100-{0.2,0.4,0.6,0.8}}
         soft_terms-w2v.ql-mlev-early-soft-none-{5-{1,10,100}-0.0,5-100-{0.2,0.4,0.6,0.8}}
         soft_terms-w2v.ql-mrel_mlev-early-soft-none-5-100-0.0
         soft_terms-{w2v.{ql,googlenews},glove.{enwiki_gigaword5,common_crawl},fasttext.enwiki}-mrel-early-hard-none-5-100-0.0
         soft_terms-w2v.ql-mrel-early-none-none-5-100-0.0)

echo Preparing the models.
parallel --halt=2 --bar --jobs=1 --line-buffer 'python3 __main__.py {} dry_run' ::: "${CONFIGS[@]}"

echo Running the evaluation.
for YEAR in dev 2016 2017; do 
  (echo config,MAP,AvgRec,MRR
  parallel --halt=2 --jobs=$(($WORKERS_NUM / 3)) -- '
    set -e
    RESULTS="$(python3 __main__.py {} '$YEAR')"
    read TEST_DIRNAME GOLD_BASE_FNAME BASE_OUTPUT_FNAME < <(echo $RESULTS)
    cd $TEST_DIRNAME
    python2 _scorer/ev.py $GOLD_BASE_FNAME $BASE_OUTPUT_FNAME | tee $BASE_OUTPUT_FNAME.score \
      | sed -n -r "/^ALL SCORES:/{s/^ALL SCORES:/{}/;s/\t/,/g;s/^([^,]*(,[^,]*){3,3}),.*/\1/;p}"
    ' ::: "${CONFIGS[@]}") | tee results-${YEAR}_unsorted.csv | sort -r -t, -k 2 >results-${YEAR}.csv &
done
wait
