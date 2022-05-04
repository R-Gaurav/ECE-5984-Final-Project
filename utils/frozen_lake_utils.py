#
# Implements the Utility code for the FrozenLake RL task.
#

import _init_paths
import numpy as np

import consts.exp_consts as epc

class FrozenLakeUtils(object):
  def __init__(self, shape):
    """
    Args:
      shape <str>: One of "4x4", "8x8" - The shape of the FrozenLake env grid.
    """
    self._shape = epc.FL_SHAPE[shape]

  def get_one_hot_encoding(self, state):
    """
    Returns the one hot encoded vector for the passed `state`.

    Args:
      state <int>: The state of the environment.

    Returns:
      numpy.ndarray
    """
    return np.eye(np.prod(self._shape))[state]

  def choose_pblt_based_action(self, t, p):
    """
    The agent chooses an action out of available options based on the input
    probability `p`.

    Args:
      t <float>: Time variable required for Nengo Node.
      p <numpy.ndarray>: An array of floating point probabilities.

    Returns:
      []
    """
    action_choice = np.random.choice(np.arange(epc.FL_ACTION_SPACE), p=p)
    result = [-1, -1, -1, -1]
    result[action_choice] = 1
    return result

  def a2c_error_function(self, t, x):
    """
    Implements the Advantage Actor Critic Error.

    Args:
      t <float>: Time variable required for Nengo Node.
      x <numpy.ndarray>: Values required to compute the A2C error.

    Returns:
      numpy.ndarray
    """
    delta, chosen, prob = x[0], x[1:5], x[5:9]
    error = np.where(chosen>0, delta*(1-prob), -delta*prob)
    return error
