import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from libs.memory import PrioritizedReplayMemory


CAPCITY = int(5e4)   # Prioritized Replay Capacity
BATCH_SIZE = 64      # Batch Size
GAMMA = 0.99         # Discount
TAU = 1e-3           # Soft update of target network
LR = 5e-4            # Learning rate
UPDATE_FREQUENCY = 4 # Frequency for training network


class Agent():

    def __init__(self, state_size, state_type, action_size, q_local, q_target, enable_double=False, random_state=42):
        """Initialize an Agent object.
        
        Arguments:
            state_size {int} -- Dimension of state space
            state_type {str} -- type of state space. Options: discrete|pixels
            action_size {int} -- Dimension of action space
            q_local {nn.Module} -- Local Q network
            q_target {nn.Module} -- Target Q network
        
        Keyword Arguments:
            model_name {str} -- which model to chose; DQN or DuelDQN (default: {'DQN'})
            enable_double {bool} -- whether to enable double DQN (default: {False})
            random_state {int} -- seed for torch random number generator (default: {42})
        """


        # Settings
        self.enable_double = enable_double
        self.state_type = state_type

        # Save action & state space
        self.state_size = state_size
        self.action_size = action_size

        # Whether to use GPU or CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Bookkeeping for loss & entropy
        self.loss_list = []
        self.entropy_list = []

        # Get the neural network based on model_name argument
        self.q_local = q_local.to(self.device)
        self.q_target = q_target.to(self.device)

        # Get an optimizer for the local network
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=LR)

        # Visualize network
        print(self.q_local)

        # Prioritized Experience Replay
        self.memory = PrioritizedReplayMemory(CAPCITY)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def local_prediction(self, state):
        """Predict Q values for given state using local Q network
        
        Arguments:
            state {array-like} -- Dimension of state space
        
        Returns:
            [array] -- Predicted Q values for each action in state
        """

        pred = self.q_local(
            Variable(torch.FloatTensor(state)).to(self.device)
        )
        pred = pred.data[0] if self.state_type == 'continuous' else pred.data
        return pred

    def step(self, state, action, reward, next_state, done):

        # Get the timporal difference (TD) error for prioritized replay
        self.q_local.eval()
        self.q_target.eval()        
        with torch.no_grad():

            # Get old Q value. Not that if continous we need to account for batch dimension
            old_q = self.local_prediction(state)[action]

            # Get the new Q value.
            new_q = reward
            if not done:
                new_q += GAMMA * torch.max(
                    self.q_target(
                        Variable(torch.FloatTensor(next_state)).to(self.device)
                    ).data
                )
            td_error = abs(old_q - new_q)

        self.q_local.train()
        self.q_target.train()

        # Save experience in replay memory
        self.memory.add(td_error.item(), (state, action, reward, next_state, done))

        # Learn every UPDATE_FREQUENCY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_FREQUENCY
        if self.t_step == 0:

            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:                
                experiences, idxs, is_weight = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, idxs, is_weight)

    def act(self, state, eps=0.):
        """Determine agent action based on state
        
        Arguments:
            state {array_like} -- current state
        
        Keyword Arguments:
            eps {float} -- epsilon, for epsilon-greedy action selection (default: {0.})
        
        Returns:
            [int] -- action to take by agent
        """

        
        # Epsilon-greedy action selection
        if random.random() > eps:

            # Get predictions from local q network
            self.q_local.eval()
            with torch.no_grad():
                action_values = self.local_prediction(state)
            self.q_local.train()

            return np.argmax(action_values.cpu().data.numpy()).astype(np.int32)
        else:

            # Random action
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, idxs, is_weight):
        """Update network given samples taken from prioritized replay memory
        
        Arguments:
            experiences {[type]} -- tuple of (s, a, r, s', done) tuples
            idxs {list} -- list of sample indexes
            is_weight {np.array} -- importance sampling weights
        """

        # Unpack experiences
        states, actions, rewards, next_states, dones = experiences

        # Convertions
        states = Variable(torch.Tensor(states)).float().to(self.device)
        next_states = Variable(torch.Tensor(next_states)).float().to(self.device)
        actions = torch.LongTensor(actions).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        is_weight = torch.FloatTensor(is_weight).unsqueeze(1).to(self.device)

        # Use DoubleDQN or DQN to get maximum Q for next state
        if self.enable_double:
            q_local_argmax = self.q_local(next_states).detach().argmax(dim=1).unsqueeze(1)
            q_targets_next = self.q_target(next_states).gather(1, q_local_argmax).detach()

        else:
            q_targets_next = self.q_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Get Q values for chosen action
        predictions = self.q_local(states).gather(1, actions)

        # Calculate TD targets
        targets = (rewards + (GAMMA * q_targets_next * (1 - dones)))

        # Update priorities
        errors = torch.abs(predictions - targets).data.cpu().numpy()
        for i in range(len(errors)):
            self.memory.update(idxs[i], errors[i])

        # Get the loss, using importance sampling weights
        loss = (is_weight * nn.MSELoss(reduction='none')(predictions, targets)).mean()

        # Run optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update book-keeping lists
        with torch.no_grad():
            self.loss_list.append(loss.item())

        # update target network
        self.soft_update(self.q_local, self.q_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """Update the target model parameters (w_target) as follows:
        w_target = TAU*w_local + (1 - TAU)*w_target

        Arguments:
            local_model {PyTorch model} -- local model to copy from
            target_model {PyTorch model} -- torget model to copy to
            tau {float} -- interpolation parameter
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)