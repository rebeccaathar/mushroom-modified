from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.utils.table import Table
import numpy as np

class QLearning(TD):
    """
    Q-Learning algorithm.
    "Learning from Delayed Rewards". Watkins C.J.C.H.. 1989.
    
    """
    def __init__(self, mdp, policy, learning_rate):
        
        #Observation space/ Action space 
        Q = Table((5184,64))
      
        super().__init__(mdp, policy, Q, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):

        q_current = self.Q[state, action]

        q_next = np.max(self.Q[next_state, :]) if not absorbing else 0.

        self.Q[state, action] = q_current + self._alpha(state, action) * (
            reward + 0.9 * q_next - q_current)

