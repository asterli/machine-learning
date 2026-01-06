"""
This module will contain the implementation of a simple ReplayBuffer class for 
storing and sampling transitions of a Markov Decision Process.

Instruction: Implement the missing parts. Only modify this file where it says "YOUR CODE HERE".
"""
import random
from collections import deque
from utils import BatchTransition, Transition

class ReplayBuffer:
    """
    A class to represent a replay buffer for storing transitions.

    Attributes:
        buffer (deque): A deque to store transitions with a maximum length. 
                        When the capacity is reached, we drop earliest items from the buffer 
                        (i.e., "Last In Last Out" like a stack).
    """
    def __init__(self, capacity: int = 10000) -> None:
        """
        Initialize the replay buffer.

        Args:
            capacity (int): The maximum number of transitions to store. Defaults to 10000.
        """
        self.buffer = deque(maxlen=capacity) # DO NOT MODIFY

    @property
    def size(self) -> int:
        """
        Get the current size of the buffer.

        Returns:
            int: The number of transitions in the buffer.
        """
        return len(self.buffer) # DO NOT MODIFY
    
    def push(self, transition: Transition) -> None:
        """
        Add a transition to (the end of) buffer.

        Args:
            transition (Transition): The transition to add.
        """
        ### YOUR CODE HERE ###
        self.buffer.append(transition)
        ### END OF YOUR CODE ###

    def sample(self, batch_size: int) -> BatchTransition:
        """
        Sample a batch of transitions from the buffer without replacement.

        Args:
            batch_size (int): The number of transitions to sample from the replay buffer.
                            Must be greater than 0.

        Returns:
            BatchTransition: A batch of sampled transitions. If the number of transitions 
                            in the buffer is less than the batch size, return all the transitions 
                            in the buffer (as a BatchTransition).

        Notes:
            - Sampling should be done *without replacement*, meaning each transition can only be 
              selected once per batch
        """
        ### YOUR CODE HERE ###
        if batch_size <= 0:
            raise ValueError("batch size must be greater than 0")
        if len(self.buffer) < batch_size:
            transitions = list(self.buffer)
        else:
            transitions = random.sample(self.buffer, batch_size)
        return BatchTransition.from_list(transitions)
        ### END OF YOUR CODE ###
           
    def clear(self) -> None:
        """
        Clear the buffer.
        """
        self.buffer.clear() # DO NOT MODIFY

