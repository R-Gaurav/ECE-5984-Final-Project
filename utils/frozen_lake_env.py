#
# This file implements the Frozen Lake Environment.
#
import gym
import numpy as np

class FrozenLakeEnv(object):
  def __init__(self, shape="4x4"):
    """
    Args:
      shape <str>: One of "4x4" or "8x8" - Frozen Lake Env Grid Shape.
    """
    self._shape = shape
    self._env = gym.make("FrozenLake-v1", map_name=shape, is_slippery=False)

  def act_on_env(self, t, a):
    """
    Agent acts on the environment; returns updated state info and reward, i.e.
    a_t, r_{t+1}, s_{t+1}.

    Args:
      t <float>: The time variable required by the Nengo Node.
      a <numpy.ndarray>: Discrete action taken by the agent - binary
                         vector.
    """
    if (int(t*1000) == 1): # When starting the simulation, reset the env.
      self._env.reset()

    assert np.sum(np.array(a)==1) == 1 # Make sure only one action is taken.

    a = np.argmax(a)
    next_state, reward, done, _ = self._env.step(a)
    if done == True:
      self._env.reset()

    return next_state, reward, done
