Future Features
===============

On this page, we maintain a list of planned future features of ``bayesim``. They're listed under a variety of categories, in loosely descending order of priority within each.


Visualization
-------------
* visualizing model uncertainty on parameter grid

* visualizing measured vs. simulated data with experimental and model uncertainty

* option to generate a plot of which particular observed data was used in an inference run


Other New Capabilities
----------------------
* option for entropy-based thresholding as well as `th_pm` and `th_pv` in :meth:`.run` function

* capability to pass a DataFrame objects to :meth:`.attach_model` and :meth:`.attach_observations` functions instead of just filepaths

* multiple output variables

* completely "closed-loop" version of :meth:`.run` function that does subdivision as well (need a way to save a handle to a model function for running new simulations)

* interpolation of simulated data

* alternative error models

* alternative initial sampling states other than grids


Efficiency/Speedup
--------------------
* speed up :meth:`list_model_pts_to_run` function

* do some more code profiling to identify other particularly slow functions/components

* more parallelism generally


Interfaces
----------
* finish developing command-line interface

* build a GUI!
