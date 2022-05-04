#
# Critic is supposed to return the future value of the current state, i.e. the
# expected return you will get in the future given that you are in the current
# state.
#
# The RL task is simple where the agent moves around in a circle and gets a
# reward whenever it is at a certain point in the circle.
#

class Critic(object):

  def __init__(self):
    self._theta = 0 # Initial angle in radians at which the agent is -> 0.

  def update(self):
    self._theta += 0.01
    # Set reward = 1 for a certain section of the circle.
    reward = 1 if np.cos(self._theta)>0.95 else 0

    return np.sin(self._theta), np.cos(self._theta), reward
