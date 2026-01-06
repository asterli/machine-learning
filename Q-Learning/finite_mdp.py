"""
DO NOT MODIFY THIS FILE
"""
import numpy as np
class DiscreteSpace:
    def __init__(self, n: int):
        self.n = n # number of elements in the space
    
    def sample(self):
        return np.random.randint(self.n)

class FiniteMDP:
    """
    An infinite-horizon Markov Decision Process (MDP) with finite state and action spaces following the Gymnasium API.
    
    Attributes:
        transition_kernel (np.ndarray): A 3D array where transition_kernel[s, a, s'] gives the probability of transitioning from state s to state s' using action a.
        reward_kernel (np.ndarray): A 3D array where reward_kernel[s, a, s'] gives the reward for transitioning from state s to state s' using action a.
        init_state_probs (np.ndarray, optional): A 1D array where init_state_probs[s] gives the probability of starting in state s. Defaults to None.

        action_space (gym.spaces.Discrete): The action space of the MDP.
        observation_space (gym.spaces.Discrete): The observation space of the MDP.
        state_space (gym.spaces.Discrete): (An alias for observation_space) The state space of the MDP.
        state (int, optional): The current state of the MDP. Defaults to None.
    """
    def __init__(self, transition_kernel, reward_kernel, init_state_probs=None):
        super(FiniteMDP, self).__init__()
        self.transition_kernel = transition_kernel
        self.reward_kernel = reward_kernel
        self.init_state_probs = init_state_probs

        self.n_states, self.n_actions, _ = transition_kernel.shape
        self.action_space = DiscreteSpace(self.n_actions)
        self.observation_space = DiscreteSpace(self.n_states)
        self.state = None

    @property
    def state_space(self):
        return self.observation_space
    
    def reset(self):
        """
        Resets the environment to an initial state.
        Returns:
            (state, info)
        """
        self.state = np.random.choice(self.n_states, p=self.init_state_probs)
        return self.state, {}

    def step(self, action):
        """
        Take an action in the environment and return the result.

        Args:
            action (int): The action to be taken.

        Returns:
            - state (int): The next state.
            - reward (float): The reward for taking the action.
            - terminated (bool): Whether the episode is terminated. Always False in this case.
            - truncated (bool): Whether the episode was truncated. Always False in this case.
            - info (dict): Additional information.
        """
        next_state = np.random.choice(self.n_states, p=self.transition_kernel[self.state, action])
        reward = self.reward_kernel[self.state, action, next_state]
        self.state = next_state
        return self.state, reward, False, False, {}
# DO NOT MODIFY THIS FILE

