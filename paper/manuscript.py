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


p.set_title('Byesim: a tool for fast device characterization with Bayesian inference')

p.add_abstract_sentence(r'''abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract abstract ''')

p.new_section('Introduction')

#p.s('Example of referencens as seen in ' + p.cite('fugallo2014thermal') + r''' and this refers to fig ''' + p.ref(fig_1 + 'a') + r'''referes the Eq. ''' + p.ref(nn1),br=1)

#p.add_equation(r'''E=mc^2 ''',label=nn1)

#p.add_figure(filename='figure_1',caption='',label=fig_1,center_page = True)


p.new_section('Model')
#p.s('Figure 1: high-level flowchart')

p.new_section('Software Architecture')
p.s('Figure 1: detailed workflow (including iteration to update posterior)')

p.add_figure(filename='figure_1',caption='(a) Scheme (b) Probability',center_page = False,label=fig_1)
p.add_figure(filename='figure_2',caption='Bayesian workflow',center_page = False,label = fig_2)

p.new_section('Example with diode - validation')
p.s('Figure 2: diode fit stuff')
p.add_figure(filename='figure_3',caption='Ideal diode',center_page = False,label = fig_3)

p.new_section('Example with real data')
p.s('Figure 4: fitting real data (maybe resistive diode?)')
p.add_figure(filename='figure_4',caption='Real data',center_page = False,label=fig_4)

con = p.new_section('Discussion')

p.new_section('Conclusions')


p.new_section('Acknowledgements')


p.new_section('Appendix')
p.s('include minimal code to run diode example')

p.tex()
p.make('Manuscript',show=True )
