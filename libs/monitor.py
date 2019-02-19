import time
from collections import deque
import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns


def environment_initialize(env, brain_name):
    """Initialize environment and return state
    
    Arguments:
        env {UnityEnvironment} -- Unity evironment
        brain_name {str} -- unity brain name
    
    Returns:
        [array-like] -- Initial state of environment
    """

    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    return state


def environment_step(env, brain_name, action):
    """Perform action in environment.
    
    Arguments:
        env {UnityEnvironment} -- Unity evironment
        brain_name {str} -- unity brain name
        action {int} -- action ID to take in environment
    
    Returns:
        [tuple] -- tuple of type (state, rewards, done)
    """

    env_info = env.step(action)[brain_name]
    state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    return state, reward, done


def train(env, agent, brain_name=None, 
          episodes=5000, max_steps=1000,
          eps_start=1.0, eps_end=0.001, eps_decay=0.97, 
          thr_score=13.0
):
    """Train agent in the environment

    Arguments:
        env {UnityEnvironment} -- Unity Environment
        agent {object} -- Agent to traverse environment
    
    Keyword Arguments:
        brain_name {str} -- brain name for Unity environment (default: {None})
        episodes {int} -- number of training episodes (default: {5000})
        max_steps {int} -- maximum number of timesteps per episode (default: {1000})
        eps_start {float} -- starting value of epsilon (default: {1.0})
        eps_end {float} -- minimum value of epsilon (default: {0.001})
        eps_decay {float} -- factor (per episode) used for decreasing epsilon (default: {0.97})
        thr_score {float} -- threshold score for the environment to be solved (default: {13.0})
    """


    # Scores for each episode
    scores = []

    # Last 100 scores
    scores_window = deque(maxlen=100)

    # Average scores & steps after each episode (within window)
    avg_scores = []
    
    # Best score so far
    best_avg_score = -np.inf
    
    # Loop over episodes
    time_start = time.time()
    eps = eps_start
    for i in range(1, episodes + 1):
        state = environment_initialize(env, brain_name)

        # Play an episode
        score = 0
        for _ in range(max_steps):
            action = agent.act(state, eps)
            next_state, reward, done = environment_step(env, brain_name, action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        eps = max(eps_end, eps_decay*eps) 

        # Update book-keeping variables
        scores_window.append(score)
        scores.append(score)
        avg_score = np.mean(scores_window)
        avg_scores.append(avg_score)
        if avg_score > best_avg_score:
            best_avg_score = avg_score

        # Info for user every 100 episodes
        n_secs = int(time.time() - time_start)
        print(f'Episode {i:6}\t Score: {score:.2f}\t Avg: {avg_score:.2f}\t Best Avg: {best_avg_score:.2f} Epsilon {eps:.4f}\t Memory: {len(agent.memory):6}\t Seconds: {n_secs:4}')
        time_start = time.time()

        # Check if done
        if avg_score >= thr_score:
            print(f'\nEnvironment solved in {i:d} episodes!\tAverage Score: {avg_score:.2f}')

            # Save the weights
            torch.save(
                agent.q_local.state_dict(),
                'logs/solved_{}_{}.pth'.format(
                    agent.model_name,
                    'double' if agent.enable_double else 'single'
                )
            )

            # Create plot of scores vs. episode
            _, ax = plt.subplots(1, 1, figsize=(7, 5))
            sns.lineplot(range(len(scores)), scores, label='Score', ax=ax)
            sns.lineplot(range(len(avg_scores)), avg_scores, label='Avg Score', ax=ax)
            ax.set_xlabel('Episodes')
            ax.set_xlabel('Score')
            ax.set_title('Agent: {}-{}'.format('double' if agent.enable_double else 'single', agent.model_name))
            ax.legend()
            plt.savefig('./logs/scores_{}_{}.png'.format(
            agent.model_name,
            'double' if agent.enable_double else 'single'
            ))

            break


def test(env, agent, brain_name, checkpoint):
    """Let pre-trained agent play in environment"""

    # Load trained model
    agent.qnetwork_local.load_state_dict(torch.load(checkpoint))

    # Initialize & interact in environment
    state = environment_initialize(env, brain_name)
    for _ in range(600):

        # Get action & perform step
        action = agent.act(state)
        state, _, done = environment_step(env, brain_name, action)
        if done:
            break

        # Prevent too fast rendering
        time.sleep(1 / 60.)