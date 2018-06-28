import argparse
import param_list as pml
import model as bym

def main(args=None):

 parser = argparse.ArgumentParser()


 parser.add_argument('-o','-output_variable', help='output variable')
 parser.add_argument('-name','-state_name', help='state name')

 parser.add_argument('-obs','-attach_observation', help='attach observation')
 parser.add_argument('-mod','-attach_model', help='attach model')
 parser.add_argument('-run', help='run analysis',action='store_true')
 parser.add_argument('-save_step', help='Steps between two consecutive saved probabilities ')
 parser.add_argument('-subdivide', help='subdivide',action='store_true')
 parser.add_argument('-th_pm', help='threshold in the probability')
 parser.add_argument('-th_pv', help='threshold in parameter space volume')
 parser.add_argument('-pb','-plot_probability', help='plot probability',action='store_true')
 parser.add_argument('-only_final', help='plot only final probability',action='store_true')



 args = parser.parse_args()
  
 state_name =  vars(args).setdefault('state_name','bayesim_state.h5')

 if 'o' in vars(args).keys() :

  m=bym.model(args.o)
  m.save_state() #State: only output

 if 'obs' in vars(args).keys() :
  m=bym.model(load_state=True,state_name = state_name)
  m.attach_observations(args.obs)
  m.save_state() 

 if 'mod' in vars(args).keys() :
  m=bym.model(load_state=True,state_name = state_name)
  m.attach_model(args.mod)
  m.save_state() 

 if args.run:
  m=bym.model(load_state=True,state_name = state_name)
  m.run(vars(args))
  m.save_state() 

 if args.subdivide:
  m=bym.model(load_state=True,state_name = state_name)
  m.subdivide(vars(args))
  m.save_state() 

 if args.pb:
  m=bym.model(load_state=True,state_name = state_name)
  m.plot_probability(vars(args))
  #m.save_state() 



 if ('only_final' in vars(args).keys()) and not args.pb:
  print('only_final is ignored unless -plot_probability is passed')


 if ('th_pv' in vars(args).keys() or 'th_pm' in vars(args).keys()) and not args.subdivide:
  print('th_pv and/or th_pm are ignored unless -subdivide is passed')


if __name__ == "__main__":

    main()
