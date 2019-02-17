import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """DQN: Deep Q Network"""

    def __init__(self, state_size, action_size, dense1=32, dense2=32, random_state=42):
        """
        Arguments:
            state_size {int} -- Dimension of state space
            action_size {int} -- Dimension of action space
        
        Keyword Arguments:
            dense1 {int} -- Nodes in first dense layer (default: {32})
            dense2 {int} -- Nodes in second dense layer (default: {32})
            random_state {int} -- seed for torch random number generator (default: {42})
        """

        super(DQN, self).__init__()
        torch.manual_seed(random_state)
        self.fc1 = nn.Linear(state_size, dense1)
        self.fc2 = nn.Linear(dense1, dense2)
        self.output = nn.Linear(dense2, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x


class DuelDQN(nn.Module):
    """Dueling Deep Q Network"""

    def __init__(self, state_size, action_size, dense1=32, dense2=32, random_state=42):
        """
        Arguments:
            state_size {int} -- Dimension of state space
            action_size {int} -- Dimension of action space
        
        Keyword Arguments:
            dense1 {int} -- Nodes in first dense layer (default: {32})
            dense2 {int} -- Nodes in second dense layer (default: {32})
            random_state {int} -- seed for torch random number generator (default: {42})
        """

        super(DuelDQN, self).__init__()
        torch.manual_seed(random_state)

        # Shared input
        self.fc1 = nn.Linear(state_size, dense1)

        # Value function (V)
        self.fc_v = nn.Linear(dense1, dense2)
        self.out_v = nn.Linear(dense2, 1)

        # Advantage function (A)
        self.fc_a = nn.Linear(dense1, dense2)
        self.out_a = nn.Linear(dense2, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        V = self.out_v(F.relu(self.fc_v(x)))
        A = self.out_a(F.relu(self.fc_a(x)))
        Q = V + (A - A.mean())
        return Q
