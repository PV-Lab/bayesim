from pypapers.manuscript import Manuscript as M

p = M(biblio_file ='biblio.bib', journal='prl', numberline=False, twocolumn=False)

DMSE = r'''Department of Materials Science \& Engineering, Massachusetts Institute of Technology, 77 Massachusetts Avenue, Cambridge, MA 02139, USA'''
MechE = r'''Department of Mechanical Engineering, Massachusetts Institute of Technology, 77 Massachusetts Avenue, Cambridge, MA 02139, USA'''

p.author(r'''Rachel Kurchin''',[DMSE])
p.author('Giuseppe Romano',[MechE])
p.author('Tonio Buonassisi',[MechE], email='buonassi@mit.edu')


fig_1 = p.get_new_figure_label()
fig_2 = p.get_new_figure_label()
fig_3 = p.get_new_figure_label()
fig_4 = p.get_new_figure_label()

nn1 = p.get_new_equation_label()
nn2 = p.get_new_equation_label()
nn3 = p.get_new_equation_label()
nn4 = p.get_new_equation_label()

p.alias(r'''\kappa_{\mathrm{eff}}''','keff')


p.title('Bayesim: a tool for fast device characterization with Bayesian inference')

p.abstract(r'''Target journal: Computer Physics Communications''')

p.section('Introduction')

#p.s('motivation (within PV and beyond) citing examples of our application of this idea so far (Joule paper and PVSC proceeding), add some stuff about how semiconductor device community lags behind many others in model tuning, etc.')
p.s('There are a plethora of examples across diverse scientific and engineering fields of mathematical models used to simulate the results of experimental observations. In many cases, there are input parameters to these models which are difficult to determine via direct measurement, and it is desirable to invert the numerical model -- that is, use the experimental observations to determine values of the input parameters. Bayesian inference is a fruitful framework within which to do such fitting, since the resulting posterior probability distribution over the parameters of interest can give rich insights into not just the most likely values of the parameters, but also uncertainty about these values and the potentially complicated ways in which they can covary to give equally good fits to observations.')

p.s('We have previously demonstrated the value of a Bayesian approach in using automated high-throughput temperature- and illumination-dependent current-voltage measurements (JVTi) to fit material/interface properties and defect recombination parameters in photovoltaic (PV) absorbers' + p.cite(['SnSJoule','FeBayes']) + r'''. In cases such as these, when the data model is not a simple analytical equation but rather a computationally intensive numerical model, efficient, nonredundant sampling of the parameter space when computing likelihoods becomes critical to making the fit feasible.''')

p.s("In this work, we introduce \texttt{bayesim}, a Python-based code that utilizes adaptive grid sampling to perform Bayesian parameter estimation. We discuss the structure of the code, its implementation, and provide several examples of its usage. While the authors' expertise is in the realm of semiconductor physics and thus the examples herein are drawn from that space, we also discuss the general characteristics of a problem amenable to this approach so that researchers from other fields might adopt it as well.")


p.section('Model')
p.note('Should this section titel actually just be ``Technical Background`` as well perhaps?')

p.s("Bayes' Theorem states")

p.equation(r'''P(H|E)=\frac{P(H)P(E|H)}{P(E)}''',label=nn1)

p.s(r'''where $H$ is a \textit{hypothesis} and $E$ the observed \textit{evidence}. $P(H)$ is termed the \textit{prior}, $P(E|H)$ the \textit{likelihood}, $P(H|E)$ the \textit{posterior}, and $P(E)$ is a normalizing constant. If there are $n$ pieces of evidence, this can generalize to an iterative process where''')

p.equation(r'''P(H|\{E_1,E_2,...E_n\}) = \frac{P(H|\{E_1,E_2...E_{n-1}\})P(E_n|H)}{P(E_n)}''')

p.s(r'''In a multidimensional parameter estimation problem, each hypothesis $H$ is a tuple of possible values for the fitting parameters, i.e. a point in the parameter space. In \texttt{bayesim}, likelihoods are calculated for each point using a Gaussian where the argument is the difference between observed and simulated output and the standard deviation is the sum of experimental uncertainty and model uncertainty. The experimental uncertainty is a number provided by the user, while the model uncertainty is calculated by \texttt{bayesim} and reflects the sparseness of the parameter space grid, i.e. how much simulated output changes from one grid point to another.''')

#p.figure(filename='figure_1',caption='(a) Scheme (b) Probability',center_page = False,label=fig_1)

p.s(r'''A high-level flowchart of what \texttt{bayesim} does is shown in ''' + p.ref('fig_1') + r'''a. ''')

#p.figure(filename='figure_2',caption='Bayesian workflow',center_page = False,label = fig_2)

p.section('Software Architecture and Interface')
#p.s(r'''\begin{itemize}\item description of structure of new code and workflow for using it (both Python scripting and command-line) \item figure 3 \end{itemize}''')
p.subsection('Structure')
p.s(r'''The structure of \texttt{bayeim} is shown in ''' + p.ref('fig_2') + r'''. The top-level object with which users interact is implemented in the \texttt{Model} class. The \texttt{params} module defines classes to store information about the various types of parameters (fitting parameters, experimental conditions, and measured output) while the \texttt{Pmf} class stores the probability distribution and implements the manipulations required for Bayesian updates.''')

#p.figure(filename='', caption='diagram of software structure', center_page = False, label = fig_2)


p.subsection('Interfaces')
p.s(r'''describe ways to "talk to" \texttt{bayesim}.''')

p.subsection('Dependencies')
p.note(r'''This probably makes sense to include also, right?''')

p.section('Application Examples')
p.subsection('Ideal Diode Model')

p.s(r'''\begin{itemize}\item validation example - ``observed'' data is just generated using the model and we show we can recover the correct input parameters \item figure 3 showing PMF \end{itemize}''')
p.figure(filename='figure_3',caption='Ideal diode',center_page = False,label = fig_3)

p.subsection('Example with Real Data')
p.s(r'''\begin{itemize}\item more practical example - probably fitting resistive diode to the same SnS data we used in the Joule paper \item figure 4 showing PMF (animation in ESI) and comparison of JV curves \end{itemize}''')


p.figure(filename='figure_4',caption='Real data',center_page = False,label=fig_4)

p.subsection('Maybe an example with a numerical model like PC1D')
p.subsection('Maybe a non-PV example')

p.s('thermoelectrics? TIDLS?')

p.section('Conclusions')
p.s('talk about broader applicability of approach')

p.section('Acknowledgements')


p.section('Appendix')
p.s(r'''\begin{itemize}\item include minimal code to run ideal diode example \item link to Github repo (which has installation instructions and documentation as well as list of planned future features) \end{itemize}''')

p.tex()
p.make('Manuscript',show=True )
