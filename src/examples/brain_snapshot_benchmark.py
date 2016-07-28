###
#
# 	Sample NEST script for benchmarking snapshot capabilities.
#
# 	Authon: Igor Peric (peric@fzi.de)
#
###

import numpy as np
import nest as sim
import imp
import cPickle as pickle
import logging
import pdb
import string

import hickle
import ujson

import h5py

import time

# serializing SLILiteral
from nest.pynestkernel import SLILiteral
import copy_reg
def recreate_sli(name):
    return SLILiteral(name)
def pickle_sli(sli_literal):
    return recreate_sli, (sli_literal.name,)
copy_reg.pickle(SLILiteral, pickle_sli)

def build_params_dict() :

	params = {
		'neurons': {}, # lists of parameters keyed with neuron NEST types
		'synapses': {}, # lists of parameters keyed with PyNN synapse types
		'pynn2nest': {}, # NEST names of equivalent PyNN synapse names without suffixes (_1, _2,...) 
		'nest2pynn': {},
		'model2int': {} # mapping from nest synapse name (string) to integer encoding (used for storing model name in .h5 dataset)
	}

	###### supported neurons

	model = 'iaf_cond_alpha'
	params['neurons'][model] = [ 'global_id', 'E_ex', 'vp', 'V_reset', \
		'V_th', 'tau_minus', 'I_e', 'g_L', 't_spike', 'E_L', \
		'tau_syn_ex', 'V_m', 'tau_minus_triplet', 't_ref', \
		'E_in', 'C_m', 'tau_syn_in' ]
	
	model = 'hh_cond_exp_traub'
	params['neurons'][model] = [ 'global_id', 'V_T', 'E_ex', 'vp', \
		'tau_minus', 'I_e', 'g_L', 't_spike', 'g_K', 'E_K', \
		'V_m', 'E_L', 'Act_m', 'Act_h', 'tau_syn_ex', 'tau_minus_triplet', \
		'Inact_n', 'E_Na', 'E_in', 'C_m', 'g_Na', 'tau_syn_in' ]

	model = 'aeif_cond_exp' # Nest specific model, no support for PyNN
	params['neurons'][model] = [ 'global_id', 'E_ex', 'vp', 'V_reset', 'V_peak', \
		'V_th', 'tau_minus', 'I_e', 'g_L', 't_spike', 'tau_w', 'E_L', \
		'tau_syn_ex', 'Delta_T', 'V_m', 'tau_minus_triplet', 't_ref', 'a', 'b', \
		'E_in', 'C_m', 'g_ex', 'g_in', 'w', 'tau_syn_in' ]

	###### supported synapses

	model = 'tsodyks_synapse_projection'
	params['synapses'][model] = [ 'source', 'target', 'weight', 'tau_psc', \
		'tau_rec', 'tau_fac', 'delay', 'U', 'u', 'sizeof', 'x', 'y']

	model = 'static_synapse_projection'
	params['synapses'][model] = [ 'source', 'target', 'weight', 'delay', 'receptor', 'sizeof']

	model = 'stdp_synapse_projection'
	params['synapses'][model] = [ 'source', 'target', 'weight', 'delay', 'receptor', 'sizeof', \
		'mu_plus', 'mu_minus', 'Wmax', 'alpha', 'tau_plus', 'lambda']

	###### nest_names
	params['pynn2nest']['tsodyks_synapse_projection'] = 'tsodyks_synapse'
	params['nest2pynn']['tsodyks_synapse'] = 'tsodyks_synapse_projection'

	params['pynn2nest']['static_synapse_projection'] = 'static_synapse'
	params['nest2pynn']['static_synapse'] = 'static_synapse_projection'
	
	params['pynn2nest']['stdp_synapse_projection'] = 'stdp_synapse'
	params['nest2pynn']['stdp_synapse'] = 'stdp_synapse_projection'

	# TODO: for each PyNN

	###### synapse model names to int coding (storing of string is neither efficient, nor )
	params['model2int']['tsodyks_synapse'] = 0
	params['model2int']['static_synapse'] = 1
	params['model2int']['stdp_synapse'] = 2
	# TODO: add every supported Nest synapse type with unique encoded value

	return params

def build_brain(brain_file_path):
	print('Loading brain from file: {0}'.format(brain_file_path))
	#brain_generator = imp.load_source('_dummy_brain', brain_file_path)

	import dummy_brain

	#return root_population

def stream_input(brain):

	# build recording devices
	# v_meter = sim.Create('voltmeter')

	# build input devices
	# ...

	pass



# removes suffix from name
def strip_off_suffix(model, params):
	real_model_name = None
	for model_name in params:
		if model.startswith(model_name):
			return model_name
	return real_model_name 

# accepts an ID of node as single int
def parse_node_info(node_id, params):

	model = sim.GetStatus([node_id], 'model')[0].name
	print('model: {}'.format(model))

	# search for names alike, to circumvent suffixes added by PyNN to model names
	model_name_no_suffix = strip_off_suffix(model, params)

	# check if the neuron model is supported
	if model_name_no_suffix not in params:
		print("Network contains neuron model unsupported by snapshot feature: {}.".format(model))

	# get the parameters specific for the model
	node_desc = sim.GetStatus([node_id], params[model_name_no_suffix])[0]
	
	return model, node_desc

def parse_synapse_info(synapse, params):

	model = sim.GetStatus([synapse], 'synapse_model')[0].name

	# search for names alike, to circumvent suffixes added by PyNN to model names
	model_name_no_suffix = strip_off_suffix(model, params['synapses'])

	# check if the neuron model is supported
	if model_name_no_suffix not in params['synapses']:
		print("Network contains synapse model unsupported by snapshot feature: {}.".format(model))

	# get the parameters specific for the model
	node_desc = sim.GetStatus([synapse], params['synapses'][model_name_no_suffix])[0]
	
	# convert pyNN name without suffix into Nest equivalent
	model = params['pynn2nest'][model_name_no_suffix]

	return model, node_desc

def save_snapshot(save_file_path):

	# build dict of params
	params = build_params_dict()

	# open the .h5 file for writing
	f = h5py.File(save_file_path, "w")

	# create group for storing synapses
	#h5_syn_group = f.create_group('synapses')
	# create group for storing neurons
	#h5_neuron_group = f.create_group('neurons')
	# create group for storing neurons
	h5_synapse_reference = f.create_group('synapse_references')
	
	################ PASS 1 ################

	# crawl the network to count the number of synapses of each type (for prealocation)
	# creates rolling pointers used for storage procedure
	rolling_pointer = {} # initialized to zero, used to keep the last writen index in dataset for each synapse type
	synapse_counter = {} # accumulates total number of synapses of each type
	for neuron_model in params['neurons']:
		nodes = sim.GetLeaves([0], {'model': neuron_model}) # accepts a list of subnetwork IDs... ID=0 is root subnet
		nodes = nodes[0] # convert from tupple to scalar
		for neuron in nodes:
			# get synapses for this neuron (list of IDs of postsynaptic neurons)
			ps_neurons = sim.GetConnections([neuron])
			for connection in ps_neurons:
				synapse_model = sim.GetStatus([connection], 'synapse_model')[0].name
				synapse_model_no_suffix = strip_off_suffix(synapse_model, params['synapses'])
				synapse_model_nest = params['pynn2nest'][synapse_model_no_suffix]
				# accumulate to the counters
				if synapse_model_nest not in synapse_counter:
					synapse_counter[synapse_model_nest] = 1
				else:
					synapse_counter[synapse_model_nest] = synapse_counter[synapse_model_nest] + 1
				# init rolling pointer for this synapse model
				rolling_pointer[synapse_model_nest] = 0

	print('Synapse counter: ')
	print(synapse_counter)

	# perform actual allocations
	h5_synapse_datasets = {} # stores pointers to actual datasets in .h5 file
	for synapse_type in synapse_counter:
		synapse_type_pynn = params['nest2pynn'][synapse_type]
		L_params = len(params['synapses'][synapse_type_pynn])
		L_count = synapse_counter[synapse_type]
		h5_synapse_datasets[synapse_type] = f.create_dataset('/synapses/{}'.format(synapse_type), \
			(L_count, L_params), dtype='f')

	################## PASS 2 ###############

	print('Starting pass 2...')
	# crawl through the network to assign 
	# for each supported neuron type...
	L_neuron_syn_ref = 0 # points to the last neuron->synapse reference in synapse_reference dataset 
	for neuron_model in params['neurons']:

		nodes = sim.GetLeaves([0], {'model': neuron_model}) # accepts a list of subnetwork IDs... ID=0 is root subnet
		nodes = nodes[0] # convert from tupple to scalar
		
		L_nodes = len(nodes)
		L_params = len(params['neurons'][neuron_model])

		# allocate dataset in the file for storing raw info about neurons
		neuron_storage = f.create_dataset('/neurons/{}'.format(neuron_model), \
			(L_nodes, L_params), dtype='f')

		for k in range(len(nodes)):
			neuron = nodes[k]

			# get numpy array describing the node according to schema
			model, param_list = parse_node_info(neuron, params['neurons'])
			# store data into .h5 file
			neuron_storage[k] = param_list

			# get synapses for this neuron (list of IDs of postsynaptic neurons)
			ps_neurons = sim.GetConnections([neuron])

			# store current rolling pointer, so we know from which index the neuron started
			start_rolling_pointer = rolling_pointer.copy()

			# query and store data about synapses for current neuron
			for i in range(len(ps_neurons)):
				ps_neuron = ps_neurons[i]
				synapse_model, synapse_info = parse_synapse_info(ps_neuron, params)
							
				# store synapse in appropriate place
				curr_pointer = rolling_pointer[synapse_model]
				h5_synapse_datasets[synapse_model][curr_pointer] = synapse_info
				# increment rolling pointer
				rolling_pointer[synapse_model] = curr_pointer + 1

			# store neuron->synapse references for writing at the end of the loop
			synapse_reference_buff = np.empty(shape=(0, 3), dtype='f')

			# store references from neurons to their respective synapses
			for synapse_type in rolling_pointer:
				start_idx = start_rolling_pointer[synapse_type]
				local_syn_count = rolling_pointer[synapse_type] - start_idx
				if local_syn_count > 0:
					# build the entry for synapse reference
					synapse_ref_data = np.array([params['model2int'][synapse_type], start_idx, local_syn_count])
					synapse_reference_buff = np.vstack((synapse_reference_buff, synapse_ref_data))

			# allocate dataset for synapse references of current neuron
			if synapse_reference_buff.shape[0] > 0:
				curr_neuron_syn_ref = h5_synapse_reference.create_dataset(\
				'/synapse_references/{}'.format(neuron), synapse_reference_buff.shape, \
				dtype='f', data=synapse_reference_buff)

	f.close()
	print('Done saving.')

def load_snapshot(path):
	
	# clear kernel
	sim.ResetKernel()
	# confirm that everything is resetted
	#syns = sim.GetConnections()
	#print('Connections:')
	#print(syns)

	# load the description of the network
	print('--- reading file...')
	t1 = time.clock()
	snap_file = open(path, 'r')
	brain_data = pickle.load(snap_file)
	t2 = time.clock()
	print('--- Done. It took {} seconds.'.format(t2-t1))
	#brain_data = hickle.load(snap_file)
	#brain_data = ujson.load(snap_file)
	
	#pdb.set_trace()

	print('--- creating nodes...')
	t1 = time.clock()
	# recreate nodes
	nodes = brain_data['nodes']
	for node in nodes:
		node_desc = node
		# handle node type (neuron, recording device, etc.)
		node_type = node_desc['model']
		#print('Node type: {}'.format(node_type))
		# remove irelevant parameters
		
		params = build_params_dict(node_desc)
		# handle recordables 
		# recordables = node_desc['recordables']
		my_node = sim.Create(node_type, 1, params=params)
		# for rec in recordables:
		# 	my_node.record()
	t2 = time.clock()
	print('--- Done. It took {} seconds.'.format(t2-t1))
	#sim.Create(nodes)

	# recreate connections
	print('--- creating connections...')
	connections = brain_data['connections']
	sim.DataConnect(connections)
	t1 = time.clock()
	print('--- Done. It took {} seconds.'.format(t1-t2))

def run():
	
	# build brain
	print('Building brain...')
	t1 = time.clock()
	brain_file_path = '/home/igor/hbp/brains/dummy_brain.py'
	#brain_file_path = '/home/igor/hbp/brains/braitenberg.py'
	#brain_file_path = '/Users/peric/dev/nest-brain-snapshots/braitenberg.py'
	brain = build_brain(brain_file_path)
	t2 = time.clock()
	print('Done, it took {} seconds.'.format(t2-t1))

	print('Streaming input...')
	# stream certain input into the brain, thus changing it
	stream_input(brain)
	t1 = time.clock()
	print('Done, it took {} seconds.'.format(t1-t2))
	# save snapshot to file
	print('Saving brain snapshot...')
	#snap_file_path = '/Users/peric/dev/nest-brain-snapshots/snapshot_dummy_brain.h5'
	snap_file_path = '/home/igor/hbp/brains/snapshot_dummy_brain.h5'
	save_snapshot(brain, snap_file_path)
	t2 = time.clock()
	print('Done, it took {} seconds.'.format(t2-t1))

	# instantiate new brain from saved one
	#print('Loading brain snapshot...')
	#new_brain = load_snapshot(snap_file_path)
	#t1 = time.clock()
	#print('Done, it took {} seconds.'.format(t1-t2))

	sim.PrintNetwork()

if __name__ == "__main__":
	run()