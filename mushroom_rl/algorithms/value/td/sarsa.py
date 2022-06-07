from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.utils.table import Table


class SARSA(TD):
    """
    SARSA algorithm.

    """
    def __init__(self, mdp, policy, learning_rate):
        
        # Observation space / Action space  
        Q = Table((5184,64))

        super().__init__(mdp, policy, Q, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]

        self.next_action = self.draw_action(next_state)
        q_next = self.Q[next_state, self.next_action] if not absorbing else 0.

        self.Q[state, action] = q_current + self._alpha(state, action) * (
            reward + 0.9 * q_next - q_current)

