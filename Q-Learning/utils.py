"""
DO NOT MODIFY THIS FILE
"""
from typing import List, NamedTuple
import numpy as np
from finite_mdp import FiniteMDP


# Data containers
class Transition(NamedTuple):
    """
    A class to represent a single transition in the environment.
    
    Attributes:
        state (int): The current state.
        action (int): The action taken.
        reward (float): The reward received.
        next_state (int): The next state after the action.
    """
    state: int
    action: int
    reward: float
    next_state: int


class BatchTransition(NamedTuple):
    """
    A class to represent a batch of transitions.
    
    Attributes:
        states (np.ndarray): Array of states.
        actions (np.ndarray): Array of actions.
        rewards (np.ndarray): Array of rewards.
        next_states (np.ndarray): Array of next states.
    """
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray

    def size(self) -> int:
        """
        Get the size of the batch.

        Returns:
            int: The number of transitions in the batch.
        """
        return len(self.states)
    
    @staticmethod
    def from_list(transitions: List[Transition]) -> 'BatchTransition':
        """
        Create a BatchTransition from a list of Transition objects.

        Args:
            transitions (List[Transition]): List of Transition objects.

        Returns:
            BatchTransition: A BatchTransition object.
        """
        states, actions, rewards, next_states = zip(*transitions)
        return BatchTransition(
            states=np.array(states, dtype=int),
            actions=np.array(actions, dtype=int),
            rewards=np.array(rewards, dtype=float),
            next_states=np.array(next_states, dtype=int)
        )


def generate_random_mdp(n_states, n_actions, sparse_reward=False):    
    """
    Generates a random Finite Markov Decision Process (MDP) with specified dimensions.
    Args:
        n_states (int): Number of states in the MDP
        n_actions (int): Number of actions available in each state
        sparse_reward (bool, optional): If True, creates an MDP with sparse rewards where only
            reaching the first (reward 0.005) and last state (reward 1.0) yields non-zero rewards.
            If False, generates random rewards for all state-action-nextstate transitions. 
            Defaults to False.
    Returns:
        FiniteMDP: A finite MDP object with the following properties:
            - Random transition probabilities following Dirichlet distribution
            - Reward structure based on sparse_reward parameter
            - Initial state distribution: deterministic at state 0 if sparse_reward=True,
                random Dirichlet distribution if sparse_reward=False
    Note:
        The transition probabilities are generated using a Dirichlet distribution, 
        ensuring they sum to 1 for each state-action pair.
    """
    transition_kernel = np.random.dirichlet(np.ones(n_states), size=(n_states, n_actions))

    if sparse_reward:
        reward_kernel = np.zeros((n_states, n_actions, n_states))
        reward_kernel[..., -1] = 1  # Reward of 1 for reaching the last state
        reward_kernel[..., 0] = 0.005  # Reward of 5/1000 for reaching the first state
        init_state_probs = np.zeros(n_states)
        init_state_probs[0] = 1  # Start from the first state
    else:
        reward_kernel = np.broadcast_to(np.random.rand(n_states), (n_states, n_actions, n_states))
        init_state_probs = np.random.dirichlet(np.ones(n_states))
    return FiniteMDP(transition_kernel, reward_kernel, init_state_probs)

# DO NOT MODIFY THIS FILE