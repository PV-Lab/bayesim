
Technical Background
====================

This page includes some references about Bayes' Theorem and Bayesian inference and discusses the particulars of the implementation of these ideas within ``bayesim``.

Bayes' Theorem
--------------

There are a `plethora <https://brohrer.github.io/how_bayesian_inference_works.html>`_ of `great <https://brilliant.org/wiki/bayes-theorem/>`_ `explanations <https://betterexplained.com/articles/an-intuitive-and-short-explanation-of-bayes-theorem/>`_ of Bayes' Theorem out there already, so I won't go through all the bayesics here but instead refer you to one of those linked above or any number of others you can find online or in a textbook.

Assuming you understand Bayes' Theorem to your own satisfaction at this point, let's remind ourselves of some **terminology**.

.. math:: \color{firebrick} {P(H|E)} =
 \frac{\color{darkorange} {P(H)}
 \color{darkmagenta} {P(E|H)}}
 {\color{teal} {P(E)}}

The :math:`\color{firebrick}{\mathbf{\text{posterior probability}}}` of our hypothesis :math:`H` given observed evidence :math:`E` is the result of a Bayesian update to the :math:`\color{darkorange}{\mathbf{\text{prior}}}` estimate of the probability of :math:`H` given the :math:`\color{darkmagenta}{\mathbf{\text{likelihood}}}` of observing :math:`E` in a world where :math:`H` is true and the probability of observing our :math:`\color{teal}{\mathbf{\text{evidence}}}` in the first place.

Bayesian Inference and Parameter Estimation
-------------------------------------------

.. note::
  I haven't found an online explanation of this material at a not-excessively-mathy level (I firmly believe that you don't need a lot of knowledge of mathematical terminology to understand this; it can really be done in a very visual way) so I wrote my own. If you know of another, please `send it to me <rkurchin@mit.edu>`_ and I'd be happy to link to it here!

Most of the examples used to explain Bayes' Theorem have two hypotheses to disginguish between (e.g. "is it raining?": yes or no). However, to use Bayes' Theorem for *parameter estimation*, which is the problem of interest here, we need to generalize to many more than two hypotheses, and those hypotheses may be about the values of multiple different parameters. This can make it confusing to conceptualize how to generalize the types of computations we do to estimate the probability of the answer to a yes-or-no question or a dice roll to a problem statement relevant to a more general scientific/modeling inquiry.

**Example: Kinematics**

To illustrate how we do this, let's use a simple example. Suppose we want to estimate the value of :math:`g`, the acceleration due to gravity near Earth's surface, and :math:`v_0`, the initial velocity of a vertically launched projectile (e.g. a ball tossed straight up), based on some measured data about the trajectory of the ball. We know from basic kinematics that the height of the ball as a function of time should obey (assuming that the projectile's initial height is defined as 0)

.. math:: y(t) = v_0t - \frac 12 gt^2

This function represents our **model** of the data we will measure and we can equivalently write

.. math:: M(v_0, g; t) = v_0t - \frac 12 gt^2

where we've now explicitly delineated our parameters :math:`g` and :math:`v_0` and our measurement condition :math:`t`.

Now let's suppose we make a measurement after 2 seconds of flight and find that :math:`y(2)=3`, with an uncertainty in the measurement of 0.2. What does this mean about the possible values of :math:`g` and :math:`v_0`? First, we need to interpret the uncertainty number, meaning we need to introduce an **error model**. We'll use a Gaussian distribution, a very common pattern for experimental errors in all kinds of measurements:

.. math:: P(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}

where :math:`\mu` is the **mean**, :math:`\sigma` is the **standard deviation**, and the term in front of the exponential is just a normalizing constant (to make sure that the probability distribution integrates to 1). The distribution looks like this:

.. figure:: img/720px-Normal_Distribution_PDF.png
   :align: center

   You can see the impact of the two parameters - a larger :math:`\sigma` value makes the distribution wider, while :math:`\mu` simply shifts the center. (Image from `Wikipedia <https://en.wikipedia.org/wiki/Normal_distribution>`_.)

What this means for our example is that our measurement of :math:`y(2)=3.0 \pm 0.2` is converted to a distribution of possible "true" values for :math:`y(2)`:

.. math:: P(y(2)) \propto \exp\left({-\frac{(y(2)-3)^2}{2*0.2^2}}\right)

(I'm leaving off the normalization constant for convenience.) But what we *really* want is a probability distribution over our parameters, not over the measurement value itself. Fortunately, our model function lets us do just that! We can translate our distribution over possible measured values into one over possible parameter values using the model function:

.. math::
   :label: condprob

   \begin{eqnarray}
   P(v_0, g | y(2)=3 \pm 0.2) & \propto & \exp\left({-\frac{(M(v_0,g;2)-3)^2}{2*0.2^2}}\right) \\
   & \propto & \exp\left({-\frac{(2v_0 - 2g - 3)^2}{0.08}}\right)
   \end{eqnarray}


Now we can visualize what that distribution looks like in ":math:`v_0`-:math:`g`" space:

.. figure:: img/probs_1.png
   :align: center

   On the left, the probability distribution over a wide range of possible values. On the right, zoomed in to near the true value of :math:`g` to show Gaussian spread.

Another way we might want to visualize would be in the space of what the actual trajectories look like:

.. figure:: img/trajs.png
   :align: center

   On the left, :math:`y(t)` trajectories from :math:`t=0` to :math:`t=3`. On the right, zooming in on the region indicated to see spread around `y(2)=3`.

So we can see that what the inference step did was essentially "pin" the trajectories to go through (or close to) the measured point at (*t*,*y*)=(2.0,3.0).

Now let's suppose we take another measurement, a short time later: *y(2.3)=0.1*, but with a larger uncertainty, this time of 0.5. Now we return to Bayes' Theorem - our prior distribution will be the conditional distribution from Equation :eq:`condprob` above, and the likelihood will be a new conditional distribution generated in exactly the same way but for this new data point. What does the posterior look like?

.. figure:: img/probs_2.png
   :align: center

   (Note that the axis limits are smaller than above)

As we would expect, we're starting to zero in on a smaller region. And how about the trajectories?

.. figure:: img/trajs_2.png
   :align: center

   Newly refined set of trajectories shown in red, overlaid on (paler) larger set from the previous step.

As expected, we've further winnowed down the possible trajectories. If we continued this process for more and more measurements, eventually zeroing in on the correct values with greater and greater precision.

``bayesim``'s implementation
----------------------------

Of course, when our model function isn't a simple analytical equation but rather a numerical solver of some sort, we can't evaluate it on a continuous parameter space but we instead have to discretize the space into a grid and choose points on that grid at which to simulate. This introduces a so-called "model uncertainty" proportional to the magnitude of the variation in the model output as one moves around the fitting parameter space.

This model uncertainty is calculated in ``bayesim`` at each experimental condition for each point in the parameter space as the largest change in model output from that point to any of the immediately adjacent points.
