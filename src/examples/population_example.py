import sys
import pyNN
from pyNN.nest import Population, SpikeSourcePoisson, SpikeSourceArray, IF_curr_alpha, AllToAllConnector, run, end, \
    setup
import numpy as np
from pyNN.nest.projections import Projection


def main(args):

    setup(timestep=0.1)

    random_image = np.random.rand(2,2)
    size = random_image.size


    input_population_arr = Population(random_image.size, SpikeSourceArray, {'spike_times': [0 for i in range(0, random_image.size)]})

    cell_params = {'tau_refrac': 2.0, 'v_thresh': -50.0, 'tau_syn_E': 2.0, 'tau_syn_I': 2.0}
    output_population = Population(1, IF_curr_alpha, cell_params, label="output")

    projection = Projection(input_population_arr, output_population, AllToAllConnector())
    projection.setWeights(1.0)

    input_population_arr.record('spikes')
    output_population.record('spikes')

    tstop = 1000.0

    run(tstop)

    output_population.printSpikes("simpleNetwork_output.pkl")
    input_population_arr.printSpikes("simpleNetwork_input.pkl")
    #output_population.print_v("simpleNetwork.v")
    end()

if __name__ == '__main__':
    main(sys.argv)