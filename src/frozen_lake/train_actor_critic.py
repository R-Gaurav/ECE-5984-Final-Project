#
# This file trains the actor and the critic for the FrozenLake problem. Note that
# non-slippery FrozenLake is considered.
#

import _init_paths
import argparse
import nengo
import pickle
import numpy as np

import consts.dir_consts as drc
import consts.exp_consts as epc
from utils.frozen_lake_env import FrozenLakeEnv
from utils.frozen_lake_utils import FrozenLakeUtils

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--shape", type=str, default="4x4", help="Shape of the FrozenLake Env?")
  args = parser.parse_args()

  fl_obj = FrozenLakeEnv(shape=args.shape)
  fl_utl = FrozenLakeUtils(shape=args.shape)

  with nengo.Network() as net:
    env_node = nengo.Node(lambda t, x: fl_obj.act_on_env(t, x), size_in=4, size_out=3)
    state_node = nengo.Node(None, size_in=1) # Separate the state info from env.
    reward_node = nengo.Node(None, size_in=1) # Separate reward info from env.
    done_node = nengo.Node(None, size_in=1) # Separate done info from env.

    nengo.Connection(env_node[0], state_node, synapse=None)
    nengo.Connection(env_node[1], reward_node, synapse=None)
    nengo.Connection(env_node[2], done_node, synapse=None)

    ############################################################################
    ################### C R I T I C #####################
    ############################################################################

    # Critic is supposed to evaluate the value of the state.
    ens = nengo.Ensemble(n_neurons=epc.N_SPK_NEURONS, dimensions=1)
    nengo.Connection(state_node, ens, synapse=None)

    value_node = nengo.Node(None, size_in=1)

    # Probe the value, reward, and done nodes.
    probe_value = nengo.Probe(value_node)
    probe_reward = nengo.Probe(reward_node)
    probe_done = nengo.Probe(done_node)

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

    ############################################################################
    ################## A C T O R ####################
    ############################################################################
    raw_actions_node = nengo.Node(None, size_in=4, label="Actor's raw actions")
    probe_raw_actions = nengo.Probe(raw_actions_node, synapse=None)

    def softmax(t, x):
      return np.exp(x)/np.sum(np.exp(x))
    actions_prob_node = nengo.Node(
        output=softmax, size_in=4, label="Porbability Of Agent's Actions")
    nengo.Connection(raw_actions_node, actions_prob_node, synapse=None)

    choice_node = nengo.Node(
        output=fl_utl.choose_pblt_based_action, size_in=4, size_out=4,
        label="Agent's Action Choice")
    probe_choice = nengo.Probe(choice_node, synapse=None)
    nengo.Connection(actions_prob_node, choice_node, synapse=None)

    # Connect the Action Choice to the Environment.
    nengo.Connection(choice_node, env_node, synapse=None)

    # Connect for the Actor to learn the actions from the state Ensemble.
    learn_conn_act = nengo.Connection(
        ens.neurons, raw_actions_node, transform=np.zeros((4, ens.n_neurons)),
        learning_rule_type=nengo.PES(learning_rate=1e-4, pre_synapse=epc.TAU_SLOW))

    # Implement the Advantage Actor Critic to help it increase the probability of
    # better actions, and decrease the probability of poor actions.
    def a2c_error_function(t, x):
      delta, chosen, prob = x[0], x[1:5], x[5:9]
      error = np.where(chosen>0, delta*(1-prob), -delta*prob)
      return error

    actor_error_node = nengo.Node(
        output=a2c_error_function, size_in=9, label="Actor's Error")
    nengo.Connection(value_error_node, actor_error_node[0], synapse=None)
    nengo.Connection(choice_node, actor_error_node[1:5], synapse=None)
    nengo.Connection(actions_prob_node, actor_error_node[5:9], synapse=None)
    nengo.Connection(actor_error_node, learn_conn_act.learning_rule, transform=-1,
                     synapse=epc.TAU_SLOW)

  # Run the Actor-Critic model.
  sim=nengo.Simulator(net)
  sim.run(10)

  print("Simulation done! Saving results...")
  pickle.dump(sim.data[probe_done],
              open(drc.RESULTS_DIR+"/frozen_lake_task/task_done.p", "wb"))
  pickle.dump(sim.data[probe_value],
              open(drc.RESULTS_DIR+"/frozen_lake_task/actor_value.p", "wb"))
  pickle.dump(sim.data[probe_reward],
              open(drc.RESULTS_DIR+"/frozen_lake_task/actor_reward.p", "wb"))
  pickle.dump(sim.data[probe_choice],
              open(drc.RESULTS_DIR+"/frozen_lake_task/actor_act_choice.p", "wb"))
  pickle.dump(sim.data[probe_raw_actions],
              open(drc.RESULTS_DIR+"/frozen_lake_task/actor_raw_actions.p", "wb"))
  print("Results saved! Exiting...")
