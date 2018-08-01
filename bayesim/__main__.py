import argparse
import bayesim.param_list as pml
import bayesim.model as bym

def main(args=None):
    parser = argparse.ArgumentParser()

    # required values / analysis actions
    parser.add_argument('-o', '-output_variable', help='output variable')
    parser.add_argument('-obs', '-attach_observation', help='attach observation')
    parser.add_argument('-ec_x', '-ec_x_var', help='x-axis variable for data plot')
    parser.add_argument('-mod', '-attach_model', help='attach model')
    parser.add_argument('-run', help='run analysis', action='store_true')
    parser.add_argument('-sub', '-subdivide', help='subdivide', action='store_true')

    # optional stuff / tweaking behavior
    parser.add_argument('-name', '-state_name', help='state name')
    parser.add_argument('-max_step', '-max_ec_x_step', help='largest step in x-axis EC to allow')
    parser.add_argument('-save_step', default=10, help='Steps between two consecutive saved probabilities ')
    parser.add_argument('-th_pm', default=0.8, help='threshold in the probability mass')
    parser.add_argument('-th_pv', default=0.05, help='threshold in parameter space volume')
    parser.add_argument('-prb', '-plot_probability', help='plot probability', action='store_true')


    #args = {k:v for k,v in vars(parser.parse_args()).items() if not vars(parser.parse_args())[k]==None}
    args = parser.parse_args()
    state_name =  vars(args).setdefault('state_name','bayesim_state.h5')

    # define output variable (eventually allow multiple)
    #if 'o' in vars(args).keys():
    if not vars(args)['o']==None:
        if not vars(args)['ec_x']==None:
            m = bym.model(output_var=args.o, ec_x_var=args.ec_x)
        else:
            m=bym.model(output_var=args.o)
        m.save_state()

    # attach observed data
    #if 'obs' in vars(args).keys():
    if not vars(args)['obs']==None:
        m=bym.model(load_state=True, state_name=state_name)
        if not vars(args)['max_step']==None:
            m.attach_observations(fpath=args.obs, max_ec_x_step=args.max_step)
        else:
            m.attach_observations(fpath=args.obs)
        m.save_state()

    # attach modeled data
    #if 'mod' in vars(args).keys():
    if not vars(args)['mod']==None:
        m=bym.model(load_state=True, state_name=state_name)
        m.attach_model(fpath=args.mod, mode='file')
        m.calc_model_errors()
        m.save_state()

    # run inference
    if args.run:
        m=bym.model(load_state=True, state_name=state_name)
        m.run(**vars(args))
        m.save_state()

    # subdivide the grid
    if args.sub:
        m=bym.model(load_state=True, state_name=state_name)
        m.subdivide(**vars(args))
        m.save_state()

    # plot probabilities
    if args.prb:
        m=bym.model(load_state=True, state_name=state_name)
        m.visualize_probs(save_file=True)


if __name__ == "__main__":

    main()
