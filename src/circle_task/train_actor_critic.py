#
# This file trains the actor and the critic.
#

import _init_paths
import nengo
import pickle
import numpy as np

import consts.dir_consts as drc
import consts.exp_consts as epc
from environment import Environment

env = Environment()
with nengo.Network() as net:
  env_node = nengo.Node(lambda t, x: env.update(x), size_in=2, size_out=3)

  state_node = nengo.Node(None, size_in=2) # Separate the state info from env.
  nengo.Connection(env_node[:2], state_node, synapse=None)
  reward_node = nengo.Node(None, size_in=1)
  nengo.Connection(env_node[2], reward_node, synapse=None)

  ##############################################################################
  ################    C R I T I C    ##################
  ##############################################################################
  # Encode the state info through an Ensemble of neurons.
  ens = nengo.Ensemble(n_neurons=epc.N_SPK_NEURONS, dimensions=2)
  nengo.Connection(state_node, ens, synapse=None)

  value_node = nengo.Node(None, size_in=1)

  probe_value = nengo.Probe(value_node)
  probe_reward = nengo.Probe(reward_node)

  # Compute the value error.
  value_error_node = nengo.Node(None, size_in=1)
  nengo.Connection(
      value_node, value_error_node, transform=-epc.DISCOUNT, synapse=epc.TAU_FAST)
  nengo.Connection(
      value_node, value_error_node, synapse=epc.TAU_SLOW)
  nengo.Connection(
      reward_node, value_error_node, transform=-1, synapse=epc.TAU_SLOW)

  # Create the learning connection to learn the value from the state Ensemble.
  learn_conn_crt = nengo.Connection(
      ens.neurons, value_node, transform=np.zeros((1, ens.n_neurons)),
      learning_rule_type=nengo.PES(learning_rate=1e-4, pre_synapse=epc.TAU_SLOW))
  nengo.Connection(value_error_node, learn_conn_crt.learning_rule, synapse=None)

  ##############################################################################
  ################    A C T O R    ##################
  ##############################################################################
  raw_actions_node = nengo.Node(None, size_in=2, label="Actor's Raw Actions")

  def softmax(t, x):
    return np.exp(x)/np.sum(np.exp(x))
  actions_prob_node = nengo.Node(
      output=softmax, size_in=2, label="Probability Of Agent's Actions")
  nengo.Connection(raw_actions_node, actions_prob_node, synapse=None)

  def choice_func(t, x):
    # Choose action with probability `x` (from the `action_prob_node`).
    action_choice = np.random.choice(np.arange(2), p=x)
    result = [-1, -1]
    result[action_choice] = 1 # Set the choice of the action as 1.0.
    return result

  choice_node = nengo.Node(
      output=choice_func, size_in=2, size_out=2, label="Agent's Action Choice")
  nengo.Connection(actions_prob_node, choice_node, synapse=None)

  # Connect the Action Choice to the Environment.
  nengo.Connection(choice_node, env_node, synapse=None)

  # Connection for the Actor to learn the actions from the state Ensemble.
  learn_conn_act = nengo.Connection(
      ens.neurons, raw_actions_node, transform=np.zeros((2, ens.n_neurons)),
      learning_rule_type=nengo.PES(learning_rate=1e-4, pre_synapse=epc.TAU_SLOW))

  # Implement Advantage Actor Critic to help it increase the probaility of
  # better actions, and decrease the probability of poor actions.
  def a2c_error_function(t, x):
    delta, chosen, prob = x[0], x[1:3], x[3:5]
    error = np.where(chosen>0, delta*(1-prob), -delta*prob)
    return error

  actor_error_node = nengo.Node(
      output=a2c_error_function, size_in=5, label="Actor's Error")
  nengo.Connection(value_error_node, actor_error_node[0], synapse=None)
  nengo.Connection(choice_node, actor_error_node[1:3], synapse=None)
  nengo.Connection(actions_prob_node, actor_error_node[3:5], synapse=None)
  nengo.Connection(actor_error_node, learn_conn_act.learning_rule, transform=-1,
                   synapse=epc.TAU_SLOW)

  probe_actor_choice = nengo.Probe(choice_node)
  probe_actor_aprobs = nengo.Probe(actions_prob_node)


# Run the Actor-Critic model.
sim = nengo.Simulator(net)
sim.run(600) # Execute for 600 seconds.

print("Saving Results...")
pickle.dump(sim.data[probe_value],
            open(drc.RESULTS_DIR+"/circle_task/full_critic_value.p", "wb"))
pickle.dump(sim.data[probe_reward],
            open(drc.RESULTS_DIR+"/circle_task/full_env_rewards.p", "wb"))
pickle.dump(sim.data[probe_actor_choice],
            open(drc.RESULTS_DIR+"/circle_task/full_actor_choices.p", "wb"))
pickle.dump(sim.data[probe_actor_aprobs],
            open(drc.RESULTS_DIR+"/circle_task/full_actor_action_probs.p", "wb"))
pickle.dump(sim.trange(),
            open(drc.RESULTS_DIR+"/circle_task/sim_trange.p", "wb"))
print("Results saved! Exiting...")
