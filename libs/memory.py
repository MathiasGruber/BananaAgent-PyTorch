import random
import numpy as np
from libs.sumtree import SumTree


class PrioritizedReplayMemory:  
    '''
    Implementation of prioritized experience replay. Adapted from:
    https://github.com/rlcode/per/blob/master/prioritized_memory.py
    '''

    def __init__(self, capacity):
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

        self.tree = SumTree(capacity)
        self.capacity = capacity

    def __len__(self):
        """Number of samples in memory
        
        Returns:
            [int] -- samples
        """

        return self.tree.n_entries

    def _get_priority(self, error):
        """Get priority based on error
        
        Arguments:
            error {float} -- TD error
        
        Returns:
            [float] -- priority
        """

        return (error + self.e) ** self.a

    def add(self, error, sample):
        """Add sample to memory
        
        Arguments:
            error {float} -- TD error
            sample {tuple} -- tuple of (state, action, reward, next_state, done)
        """

        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        """Sample from prioritized replay memory
        
        Arguments:
            n {int} -- sample size
        
        Returns:
            [tuple] -- tuple of ((state, action, reward, next_state, done), idxs, is_weight)
        """

        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if p > 0:
                priorities.append(p)
                batch.append(data)
                idxs.append(idx)

        # Calculate importance scaling for weight updates
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)

        # Paper states that for stability always scale by 1/max w_i so that we only scale downwards
        is_weight /= is_weight.max()

        # Extract (s, a, r, s', done)
        batch = np.array(batch).transpose()
        states = np.vstack(batch[0])
        actions = list(batch[1])
        rewards = list(batch[2])
        next_states = np.vstack(batch[3])
        dones = batch[4].astype(int)

        return (states, actions, rewards, next_states, dones), idxs, is_weight

    def update(self, idx, error):
        """Update the priority of a sample
        
        Arguments:
            idx {int} -- index of sample in the sumtree
            error {float} -- updated TD error
        """

        p = self._get_priority(error)
        self.tree.update(idx, p)
