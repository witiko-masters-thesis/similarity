This directory contains a Jupyter notebook `map-density_plots.ipynb` that
was used to produce the MAP-density plots.

To reproduce the results, install the required Python packages in a Python 3
virtualenv:

    $ pip install -U pip
    $ pip install -r requirements.txt

download the SemEval-2016/2017 Task 3 datasets:

    $ make -C ../datasets

and run the main script:

    $ jupyter-notebook map-density_plots.ipynb

You should now be able to rerun all the cells in the notebook and produce the
MAP-density plots.

Note that the plot data are extracted from the log files that are distributed
contained in the repository. That means that no models are actually built and
the plotting is very fast. However, if you wish to truly reproduce our results,
you should rebuild the models first. Consult the `README.md` file in the
directory above for instructions on how to rebuild the models.
