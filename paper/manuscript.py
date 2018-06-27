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

p.add_abstract_sentence(r'''Achieving low thermal conductivity and good electrical properties is a crucial condition for thermal energy harvesting materials. Nanostructuring offers a very powerful tool to address both requirements: in nanostructured materials, boundaries preferentially scatter phonons compared to electrons. The search for low-thermal conductivity nanostructures is typically limited to materials with simple crystal structures, such as silicon, because of the complexity arising from modeling branch- and wave vector-dependent nanoscale heat transport. Using the phonon mean-free-path (MFP) dependent Boltzmann transport equation, a model that overcomes this limitation, we compute thermal transport in 75 nanoporous Half Heusler compounds for different pore sizes. We demonstrate that the optimization of thermal transport in nanostructures should take into account both bulk thermal properties and geometry-dependent phonon suppression, two aspects that are typically engineered separately. In fact, our work predicts that, given a set of bulk materials and a system geometry, the ordering of the bulk thermal conductivity does not necessarily align with that of the nanostructure. We show that what dictates thermal transport is the interplay between the bulk MFP distribution and the material's characteristic length. Finally, we derive a thermal transport model than enables fast systems screening within large bulk material repositories and a given geometry. Our study motivates the need for a holistic approach in engineering thermal transport and provides a method for high-throughput materials discovery.''')

con = p.get_new_section_id('Introduction')

p.s('Example of referencens as seen in ' + p.cite('fugallo2014thermal') + r''' and this refers to fig ''' + p.ref(fig_1 + 'a') + r'''referes the Eq. ''' + p.ref(nn1),br=1)

p.add_equation(r'''E=mc^2 ''',label=nn1)

p.add_figure(filename='figure_1',caption='',label=fig_1,center_page = True)


con = p.get_new_section_id('Model')


con = p.get_new_section_id('Software Architecture')


con = p.get_new_section_id('Example with diode')
con = p.get_new_section_id('Example with real data')


con = p.get_new_section_id('Discussion')
con = p.get_new_section_id('Conslusions')


con = p.get_new_section_id('Acknowledgements')
p.s(r'''Research supported as part of the Solid-State Solar-Thermal Energy Conversion Center (S3TEC), an Energy Frontier Research Center funded by the US Department of Energy (DOE), Office of Science, Basic Energy Sciences (BES), under Award DESC0001.''')


p.tex()
p.make('Manuscript',show=True )



