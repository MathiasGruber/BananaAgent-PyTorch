# Double-Duel-Deep-Q-Network in PyTorch
This model is developed as a solution to Project 1 of Udacity Deep Reinforcement Learning Nanodegree.

# Installation
Install the package requirements for this repository
```
pip install -r requirements.txt
```

## Banana Environment
The agent was developed specifically to solve a banana collection environment developed in Unity, which can be downloaded from the following locations. The objective in the banana environment is gor an agent to navigate and collect yellow bananas (+1 reward) while avoiding blue bananas (-1 reward). Download your specific environment and unpack it into the `./env_unity/` folder in this repo:

Environment with discrete state space (37 dimensions):
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Environment with pixel state space.
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

In both versions of the environment, the agent has an action space with four discrete actions; 
* `0`: forward
* `1`: backwards
* `2`: left
* `3`: right

The environment is considered solved when the agent collect an average score of 13 bananas over 100 consecutive episodes.

# Repository Structure
* `libs/agents.py`: A DQN agent, which by default is configured to be a double dueling DQN.
* `libs/models.py`: PyTorch models used by the DQN agent
* `libs/memory.py`: Prioritized experience replay, using sum-tree as defined in `libs/sumtree.py`
* `libs/monitor.py`: Functionality for training/testing the agent and interacting with the environment
* `main.py`: Main command-line interface for training & testing the agent

# Training the Agent
For training the agent on the discrete state space, the model can be run by using one of the following (only tested on windows!):
```
python main.py --environment env_unity/DiscreteBanana/Banana.exe --model_name DQN
python main.py --environment env_unity/DiscreteBanana/Banana.exe --model_name DuelDQN
python main.py --environment env_unity/DiscreteBanana/Banana.exe --model_name DQN --double
python main.py --environment env_unity/DiscreteBanana/Banana.exe --model_name DuelDQN --double
```

For training agent on the pixel state space, the following can be used (only tested on windows!)
```
python main.py --environment env_unity/VisualBanana/Banana.exe --model_name DQN
python main.py --environment env_unity/VisualBanana/Banana.exe --model_name DQN --double
```