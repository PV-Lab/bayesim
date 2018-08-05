
Why Bayes?
==========

.. toctree::
   :hidden:

Discuss advantages of approach over others:

- probability distribution gives much richer information than just value +/- uncertainty (which of course we can get from this too) --> shape of region of parameter space offering equally good fits, e.g. mu tau and ideal diode examples

- with grid subdivision approach, can make fitting with expensive numerical models in many dimensions actually feasible that wouldn't be with a more naïve regression approach


Why/When NOT Bayes?
-------------------

Many examples here use analytical models becaue they're fast and tractable to demonstrate on a laptop, and it will certainly work, but it's almost certainly not the most efficient approach in those cases (because you can often directly invert the equation)
