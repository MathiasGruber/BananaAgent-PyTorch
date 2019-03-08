import torch
import torch.nn as nn


class DQN(nn.Module):
    """DQN: Deep Q Network with discrete state space"""

    def __init__(self, 
        state_size,  
        state_type, 
        action_size,        
        dense_layers=[32, 32], 
        conv_filters=[128, 256, 256],
        random_state=42
    ):
        """
        Arguments:
            state_size {int or tuple} -- Dimension of state space. {int} if discrete, {(c, f, h, w)} if pixels
            state_type {str} -- type of state space. Options: discrete|pixels
            action_size {int} -- Dimension of action space
        
        Keyword Arguments:            
            dense_layers {list} -- Nodes in dense layers (default: {[32, 32]})
            conv_filters {list} -- Filters in conv layers (default: {[32, 64, 64]})
            random_state {int} -- seed for torch random number generator (default: {42})
        """

        super(DQN, self).__init__()

        # Settings
        self.state_type = state_type        

        # Set the random number generator for torch
        torch.manual_seed(random_state)

        # Either discrete state space or pixel state space
        if state_type == 'discrete':

            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(state_size, dense_layers[0])
            self.fc2 = nn.Linear(dense_layers[0], dense_layers[1])
            self.output = nn.Linear(dense_layers[1], action_size)

        elif state_type == 'continuous':

            c, _, _, _ = state_size
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv3d(c, conv_filters[0], kernel_size=(1, 3, 3), stride=(1, 3, 3))
            self.bn1 = nn.BatchNorm3d(conv_filters[0])
            self.conv2 = nn.Conv3d(conv_filters[0], conv_filters[1], kernel_size=(1, 3, 3), stride=(1, 3, 3))
            self.bn2 = nn.BatchNorm3d(conv_filters[1])
            self.conv3 = nn.Conv3d(conv_filters[1], conv_filters[2], kernel_size=(4, 3, 3), stride=(1, 3, 3))
            self.bn3 = nn.BatchNorm3d(conv_filters[2])
            self.fc1 = nn.Linear(self._get_conv_out(state_size), 512)
            #self.fc2 = nn.Linear(64, 64)
            self.output = nn.Linear(512, action_size)
            

        else:
            raise AttributeError('Unknown state space type: {}'.format(state_type))

    def forward(self, state):

        if self.state_type == 'discrete':

            x = self.relu(self.fc1(state))
            x = self.relu(self.fc2(x))
            x = self.output(x)

        elif self.state_type == 'continuous':

            x = self._cnn(state)
            x = self.relu(self.fc1(x))
            #x = self.relu(self.fc2(x))
            x = self.output(x)
            
        else:
            raise AttributeError('Unknown state space type: {}'.format(
                self.state_type
            ))

        return x

    def _cnn(self, state):
        """Convolutional part of the network
        
        Arguments:
            state {tuple} -- Dimension of state space {(c, f, h, w)}
        
        Returns:
            x -- output of the convolutional layers
        """

        x = self.relu(self.bn1(self.conv1(state)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.reshape(x.size(0), -1)

        return x

    def _get_conv_out(self, state_size):
        x = torch.rand(state_size).unsqueeze(0)
        x = self._cnn(x)
        n_size = x.data.view(1, -1).size(1)
        return n_size


class DuelDQN(nn.Module):
    """Dueling Deep Q Network"""

    def __init__(self, 
        state_size,
        state_type,
        action_size,
        dense1=32,
        dense2=32,
        random_state=42
    ):
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
        x = self.relu(self.fc1(state))
        V = self.out_v(self.relu(self.fc_v(x)))
        A = self.out_a(self.relu(self.fc_a(x)))
        Q = V + (A - A.mean())
        return Q
