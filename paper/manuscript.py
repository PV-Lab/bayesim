from pypapers.manuscript import Manuscript as M

p = M(biblio_file ='biblio.bib',journal='prl',numberline=False)

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

con = p.get_new_section_id('Introduction')

p.s('introduction')
#p.s('Example of referencens as seen in ' + p.cite('fugallo2014thermal') + r''' and this refers to fig ''' + p.ref(fig_1 + 'a') + r'''referes the Eq. ''' + p.ref(nn1),br=1)

#p.add_equation(r'''E=mc^2 ''',label=nn1)

#p.add_figure(filename='figure_1',caption='',label=fig_1,center_page = True)


con = p.get_new_section_id('Model')
p.s('Figure 1: flowchart')

con = p.get_new_section_id('Software Architecture')
p.s('Figure 2: structure of code')

con = p.get_new_section_id('Example with diode')
con = p.get_new_section_id('Example with real data')


con = p.get_new_section_id('Discussion')
p.s('discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion discussion ')

con = p.get_new_section_id('Conslusions')
p.s('conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions conclusions ')


con = p.get_new_section_id('Acknowledgements')
p.s(r'''acknowledgements acknowledgements acknowledgements acknowledgements acknowledgements acknowledgements acknowledgements acknowledgements acknowledgements acknowledgements acknowledgements acknowledgements acknowledgements acknowledgements acknowledgements acknowledgements ''')


p.tex()
p.make('Manuscript',show=True )
