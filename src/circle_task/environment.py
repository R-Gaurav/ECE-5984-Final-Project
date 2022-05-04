#
# This file implements the Environment consisting of the Actor and the Critic.
#
# The value error computed for the Critic can be reused for the Actor to find out
# the optimal action to be taken, given a state. If the Actor does an action `a`
# and it turns out better, i.e. error(t-1)/delta(t-1) > 0 => increase the chance
# of doing that action in that particular state i.e. s(t-1).
#
#       delta(actor, a) = delta(t-1) if a is the chosen action, else 0.
#
# Above equation is used to increase the chance of the Actor doing action `a`, if
# the action led to a reward, i.e. delta(t-1) > 0.
#

# TODO: Variations -> If the action `a` taken turns out to be good, then not only
# increase its probability but also decrease the probability of choosing other
# actions. Implementing Advantage Actor Critic??
#

import nengo
import numpy as np

class Environment(object):
  def __init__(self):
    self._theta = 0

  def update(self, x):
    """
    Update the 2D state.

    Args:
      x <np.ndarray>: Action taken by the Actor, according to which the state is
                      updated.
    """
    if x[0] > 0:
      self._theta += 0.01
    if x[1] > 0:
      self._theta -= 0.01

    reward = 1.0 if np.cos(self._theta)>0.95 else 0
    return np.sin(self._theta), np.cos(self._theta), reward
