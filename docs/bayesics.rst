
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

To illustrate how we do this, let's use a simple example. Suppose we want to estimate the value of :math:`g`, the acceleration due to gravity near Earth's surface, and :math:`v_0`, the initial velocity of a vertically launched projectile (e.g. a ball tossed straight up), based on some measured data bout the trajectory of the ball. We know from basic kinematics that the height of the ball as a function of time should obey (assuming that the projectile's initial height is defined as 0)

.. math:: y(t) = v_0t - \frac 12 gt^2

This function represents our **model** of the data we will measure and we can equivalently write

.. math:: M(v_0, g; t) = v_0t - \frac 12 gt^2

where we've now explicitly delineated our parameters :math:`g` and :math:`v_0` and our measurement condition :math:`t`.

Now let's suppose we make a measurement after 2 seconds of flight and find that :math:`y(2)=3`, with an uncertainty in the measurement of 0.2. What does this mean about the possible values of :math:`g` and :math:`v_0`? First, we need to interpret the uncertainty number, meaning we need to introduce an **error model**. We'll use a Gaussian distribution, a very common pattern for experimental errors in all kinds of measurements:

.. math:: P(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}

where :math:`\mu` is the **mean**, :math:`\sigma` is the **standard deviation**, and the term in front of the exponential is just a normalizing constant (to make sure that the probability distribution integrates to 1). The distribution looks like this:

.. image:: img/Normal_Distribution_PDF.png
   :align: center

(Image from `Wikipedia <https://en.wikipedia.org/wiki/Normal_distribution>`_.)

What this means for our example is that our measurement of :math:`y(2)=3.0 \pm 0.2` is converted to a distribution of possible "true" values for :math:`y(2)`:

.. math:: P(y(2)) \propto \exp\left({-\frac{(y(2)-3)^2}{2*0.2^2}}\right)

(I'm leaving off the normalization constant for convenience.) But what we *really* want is a probability distribution over our parameters, not over the measurement value itself. Fortunately, our model function lets us do just that! We can translate our distribution over possible measured values into one over possible parameter values using the model function:

.. math:: P(v_0, g | y(2)=3 \pm 0.2) \propto \exp\left(\right)

how to generalize to multiple dimensions, incorporate uncertainty, etc.



``bayesim``'s implementation
----------------------------

.. This equation is our **model** of the data. We will use Bayes' Theorem to compare **simulated data** resulting from different combinations of values of the **parameters** of the model (:math:`g` and :math`v_0`) to the

.. Let us assume we want to calculate the thermal conductivity of a bar with length L and cross-sectional area A, subjected to a difference of temperatue :math:`\Delta T`. At the hot contact (HC), there will be outgoing thermal flux :math:`\mathbf{J}(\mathbf{r})`, which is related to the effective thermal conductivity :math:`\kappa_{eff}` via Fourier's law, i.e.

.. .. math:: \kappa_{eff} = -\frac{L}{\Delta T A}\int_{HC} dS \, \mathbf{J}(\mathbf{r}) \cdot \mathbf{\hat{n}}
  :label: kappa

.. Due to the presence of classical phonon size effects, heat transport is not diffusive, and :math:`\mathbf{J}(\mathbf{r})` needs to be calculated by the Boltzmann transport equation (BTE). OpenBTE is based on the mean-free-path formulation of the BTE. The starting point of its derivation is the standard steady-state BTE in the relaxation time approximation


 .. math::
  \mathbf{v}_\lambda \cdot \nabla f_\lambda (\mathbf{r}) = \frac{1}{\tau_\lambda}\left[f^0_\lambda(T) - f_\lambda(\mathbf{r}) \right],
  :label: bte

 where :math:`\lambda` collectively describes phonon wave vector :math:`\mathbf{q}` and polarization :math:`p`, :math:`\mathbf{v}_\lambda` is the group velocity, :math:`f_\lambda(\mathbf{r})` is the non-equilibrium distribution function. The equilibrium function :math:`f_\lambda^0(\mathbf{r})` is the Bose-Einstain distribution at the effective temperature :math:`T(\mathbf{r})`, i.e.

 .. math:: f^0_\lambda(\mathbf{r})=\left(e^{\frac{\hbar \omega_\lambda}{k_B T(\mathbf{r})}} + 1 \right)^{-1},
  :label: equilibrium

 where :math:`k_B` is the Boltzmann constant and :math:`\hbar\omega_\lambda` is the phonon energy. Energy conservation requires :math:`\nabla \cdot \mathbf{J}(\mathbf{r}) = 0`, where the total phonon flux :math:`\mathbf{J}(\mathbf{r})` is defined by

 .. math:: \mathbf{J}(\mathbf{r}) = \int \hbar\omega_\lambda \mathbf{v}_\lambda f_\lambda(\mathbf{r})  \frac{d\mathbf{q}}{8\pi^3}.
  :label: thermal

 After multiplying both sides of Eq. :eq:`bte` by :math:`\hbar \omega_\lambda` and integrating over the B. Z., we have

 .. math:: \int  \frac{d\mathbf{q}}{8\pi^3} \frac{\hbar\omega_\lambda}{\tau_\lambda} \left[f_\lambda^0(T) -f_\lambda(\mathbf{r})\right] = 0.
  :label: energy

 In practice, one has to compute :math:`T(\mathbf{r})` such as Eq. :eq:`energy` is satisfied. To simplify this task, we assume that the temperatures variation are small such that the equilibrium distribution can be approximated by its first-order Taylor expansion, i.e.

 .. math:: f_\lambda^0(T) \approx f_\lambda^0(T_0) + \frac{C_\lambda}{\hbar\omega_\lambda}\left([T(\mathbf{r})-T_0 \right],
  :label: expansion

 where :math:`C_\lambda(T_0)` is the heat capacity at a reference temperature :math:`T_0`. After including Eq. :eq:`expansion` into Eq. :eq:`energy`, we have

 .. math:: T(\mathbf{r}) -T_0 = \int  \frac{d\mathbf{q}}{8\pi^3} a_\lambda \frac{\hbar \omega_\lambda}{C_\lambda}\left[f_\lambda(\mathbf{r}) - f_\lambda^0(T_0)\right],
  :label: temperature

 where

 .. math:: a_\lambda = \frac{C_\lambda}{\tau_\lambda} \left[\int  \frac{d\mathbf{q}}{8\pi^3} \frac{C_\lambda}{\tau_\lambda} \right]^{-1}.
  :label: coefficients

 The BTE under a small applied temperature gradients can be then derived after including Eqs. :eq:`temperature`-:eq:`expansion` into Eq. :eq:`bte`

 .. math::
  \tau_\lambda \mathbf{v}_\lambda \cdot \nabla f_\lambda (\mathbf{r}) +f_\lambda(\mathbf{r}) - f_\lambda^0(T_0) = \frac{C_\lambda}{\hbar \omega_\lambda}\int \frac{d\mathbf{q}'}{8\pi^3} a_\lambda' \frac{\hbar \omega_{\lambda'}}{C_{\lambda'}}\left[f_{\lambda'}(\mathbf{r}) - f_{\lambda'}^0(T_0)) \right].
  :label: bte2

 Upon the change of variable

 .. math::
  T_\lambda(\mathbf{r}) = \frac{\hbar\omega_\lambda}{C_\lambda}\left[f_\lambda(\mathbf{r})- f_\lambda^0(T_0) \right],
  :label: variable

 we obtain the temperature formulation of the BTE

 .. math:: \mathbf{F}_\lambda \cdot \nabla T_\lambda(\mathbf{r}) + T_\lambda(\mathbf{r}) - \int \frac{d\mathbf{q}'}{8\pi^3} a_{\lambda'}T_{\lambda'}(\mathbf{r}) = 0,
  :label: bte3

 where :math:`\mathbf{F}_\lambda=\mathbf{v}_\lambda \tau_\lambda`. Within this formulation, the thermal flux becomes

 .. math:: \mathbf{J}(\mathbf{r}) = \int \frac{d\mathbf{q}}{8\pi^3} \frac{C_\lambda}{\tau_\lambda} T_\lambda(\mathbf{r})  \mathbf{F}_\lambda.
  :label: thermal2


.. Finally, it is possible to show that in the case of isotropic B.Z., Eq. :eq:`bte3` can be approximated by

.. .. math:: \Lambda \mathbf{\hat{s}} \cdot \nabla T(\mathbf{r},\Lambda) + T(\mathbf{r},\Lambda) - \int_0^{\infty} d\Lambda' B_2(\Lambda) \overline{T}(\mathbf{r},\Lambda') = 0,
  :label: bte4

.. where :math:`\overline{T}=\left(4\pi \right)^{-1}\int_{4\pi}f(\Omega)d\Omega` is an angular average and

.. .. math:: B_n(\Lambda) = \frac{K_{\mathrm{bulk}}(\Lambda)}{\Lambda^n}\left[ \int_0^\infty \frac{K_{\mathrm{bulk}}(\Lambda')}{\Lambda'^n} d\Lambda'  \right]^{-1}.

.. Similarly, the thermal flux becomes

.. .. math:: \mathbf{J}(\mathbf{r}) = \int_0^{\infty} B_1(\Lambda)  <T(\mathbf{r},\Lambda) \mathbf{\hat{s}}> d\Lambda.
  :label: thermal2
