#
# This file trains the Critic to identify the value of a state.
#
# TODO: Study the effect of increasing the number of neurons, learning rate,
# spiking neuron type, synaptic time constants - TAU_FAST, TAU_SLOW, etc. on the
# critic.
# Also, try using actual delays, e.g. LMU as the delay network, making the
# ensemble sparse, by appropriately setting up the intercepts of the neurons.
#
# TODO: Try removing the synapse from the PES rule to let the current input from
# the neuron be accounted for, instead of the delayed input. See what happens.
# Remove the negative signs from the value error equation.
#

import _init_paths
import nengo
import pickle

from consts import dir_consts as drc
from consts import exp_consts as epc
from critic import Critic

env = Critic() # Create the environment with Critic only.

with nengo.Network() as net:

  # Create the Environment Node. The environment is updated every time-step of
  # the simulation. It has three outputs, first two are 2D state values, last one
  # is the reward (which is 1 or 0 depending on the agent's state).
  env_node = nengo.Node(lambda t: env.update(), size_out=3)

  # Set other nodes to grab the 2D state and reward info.
  state_node = nengo.Node(size_in=2)
  nengo.Connection(env_node[:2], state_node, synapse=None) # Get first two otpts.
  reward_node = nengo.Node(size_in=1)
  nengo.Connection(env_node[2], reward_node, synapse=None) # Get the last otpt.

  # Create the spiking network to encode the state. The default is LIF neurons.
  ens = nengo.Ensemble(n_neurons=epc.N_SPK_NEURONS, dimensions=2)
  nengo.Connection(state_node, ens, synapse=None)

  value_node = nengo.Node(None, size_in=1) # Otpt value that critic will learn.

  # Record/Probe the critic's value and agent's reward from the environment.
  probe_value = nengo.Probe(value)
  probe_reward = nengo.Probe(reward)

# Compute the error from the Critic's evaluated value for the state at time t.
# error(t) or delta(t) = (
#     reward(t) + DISCOUNT*value(state(t+1)) - value(state(t)))
#
# Since during network simulation, we cannot access future state values at t+1,
# we shift the time backwards. i.e.
# error(t-1) or delta(t-1) = (
#     reward(t-1) + DISCOUNT*value(state(t)) - value(state(t-1)))
#
# For implementation in Nengo, we need a bunch of neurons which store the state.
# And since the above error/delta involves shifting things in time, i.e. getting
# the value of the previous state, we need to figure out a way to implement the
# time delay. If we are representing the input current states through an Ensemble
# then it is possible to access the previous state by synapsing the state signal
# from the ensemble. Note that synapses delay the input signal - the magnitude
# of delay is determined by the extent of filtering done or the synapse
# time-constant. Also note that a symbolic delay of t-1 is done through synapsing,
# i.e. it is not necessarily t-1 since one time-step delay makes little sense in
# continuous domain spiking networks.

with net: # Computing the error signal.
  value_error_node = nengo.Node(size_in=1)
  nengo.Connection( # DISCOUNT*value(state(t))
      value_node, value_error_node, transform=-epc.DISCOUNT, synapse=epc.TAU_FAST)
  nengo.Connection( # value(state(t-1))
      value_node, value_error_node, synapse=epc.TAU_SLOW)
  nengo.Connection( # reward(t-1)
      reward_node, value_error_node, transform=-1, synapse=epc.TAU_SLOW)

# Above error signal has to be applied to a learning connection, which is the
# PES rule. But the error signal we have got is w.r.t. the previous time-step,
# i.e. error(t-1) instead of error(t), and the input x_i (in the PES rule) is
# the current input at time t from the input neuron. We need to delay it too,
# to conform with the delayed error signal. We can do that by applying a synaptic
# filter in the PES rule to the current input from the input neurons.

with net:
  # Connect the Ensemble neurons to the value with PES rule so as to learn the
  # weights for value prediction.
  learn_conn = nengo.Connection(
      ens.neurons, value, transform=np.zeros((1, ens.n_neurons)),
      learning_rule_type=nengo.PES(learning_rate=1e-4, pre_synapse=epc.TAU_SLOW))
  # Connect the error signal to the learning rule.
  nengo.Connection(value_error_node, learn_conn.learning_rule, synapse=None)

# Run the simulation and save the results.
sim = nengo.Simulator(net)
sim.run(100) # Execute for 100 seconds.
print("Saving Results...")
pickle.dump(sim.data[probe_value],
            open(drc.RESULTS_DIR+"/circle_task/basic_critic_value.p", "wb"))
pickle.dump(sim.data[probe_reward],
            open(drc.RESULTS_DIR+"/circle_task/enviornment_rewards.p", "wb"))
