from pypapers.manuscript import Manuscript as M

p = M(biblio_file ='biblio.bib',journal='prl',numberline=False,twocolumn=False)

MIT = 'Department of Mechanical Engineering, Massachusetts Institute of Technology, 77 Massachusetts Avenue, Cambridge, MA 02139, USA'


p.add_author('Rachel Kurchin',[MIT])
p.add_author('Giuseppe Romano',[MIT])
p.add_author('Tonio Buonassisi',[MIT],email='buonassi@mit.edu')


fig_1 = p.get_new_figure_label()
fig_2 = p.get_new_figure_label()
fig_3 = p.get_new_figure_label()
fig_4 = p.get_new_figure_label()

nn1 = p.get_new_equation_label()
nn2 = p.get_new_equation_label()
nn3 = p.get_new_equation_label()
nn4 = p.get_new_equation_label()

p.add_alias(r'''\kappa_{\mathrm{eff}}''','keff')
p.add_alias(r'''\mathbf{\hat{x}}''','x')
p.add_alias(r'''\kappa_{\mathrm{bulk}}''','kbulk')
p.add_alias(r'''K_{\mathrm{bulk}}''','Kbulk')
p.add_alias(r'''\tilde{\kappa}_{\mathrm{eff}}''','ktilde')
p.add_alias(r'''\kappa_{\mathrm{Fourier}}''','kfourier')
p.add_alias(r'''\alpha_{\mathrm{bulk}}''','abulk')
p.add_alias(r'''\mathbf{\hat{s}}''','s')


p.set_title('Bayesim: a tool for fast device characterization with Bayesian inference')

p.add_abstract_sentence(r'''Target journal: Computer Physics Communications''')

p.section('Introduction')

p.s('motivation (within PV and beyond) citing examples of our application of this idea so far (Joule paper and PVSC proceeding)')
#p.s('Example of referencens as seen in ' + p.cite('fugallo2014thermal') + r''' and this refers to fig ''' + p.ref(fig_1 + 'a') + r'''referes the Eq. ''' + p.ref(nn1),br=1)

#p.add_equation(r'''E=mc^2 ''',label=nn1)

#p.add_figure(filename='figure_1',caption='',label=fig_1,center_page = True)


p.section('Model')

s1 =  'background on Bayes theorem'
s2 =  r'''particulars about how it can be applied to fitting problems we're interested in'''
s3 = 'advantages over traditional fitting approaches'
s4 =  'figure 1'
p.itemize([s1,s2,s3,s4])




p.figure(filename='figure_1',caption='(a) Scheme (b) Probability',center_page = False,label=fig_1)

p.section('Software Architecture and Interface')
p.s(r'''\begin{itemize}\item description of structure of new code and workflow for using it (both Python scripting and command-line) \item figure 2 \end{itemize}''')

p.figure(filename='figure_2',caption='Bayesian workflow',center_page = False,label = fig_2)

p.section('Application Examples')
p.subsection('Ideal Diode Model')

p.s(r'''\begin{itemize}\item validation example - ``observed'' data is just generated using the model and we show we can recover the correct input parameters \item figure 3 showing PMF (animation in ESI) \end{itemize}''')
p.figure(filename='figure_3',caption='Ideal diode',center_page = False,label = fig_3)

p.subsection('Example with Real Data')
p.s(r'''\begin{itemize}\item more practical example - probably fitting resistive diode to the same SnS data we used in the Joule paper \item figure 4 showing PMF (animation in ESI) and comparison of JV curves \end{itemize}''')
p.figure(filename='figure_4',caption='Real data',center_page = False,label=fig_4)

p.subsection('Maybe an example with a numerical model like PC1D')
p.subsection('Maybe a non-PV example')

p.s('thermoelectrics? TIDLS?')

p.section('Conclusions')


p.section('Acknowledgements')


p.section('Appendix')
p.s(r'''\begin{itemize}\item include minimal code to run ideal diode example \item link to Github repo (which has installation instructions and documentation as well as list of planned future features) \end{itemize}''')

p.tex()
p.make('Manuscript',show=True )
