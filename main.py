import argparse
from unityagents import UnityEnvironment
from libs.agents import Agent
from libs.monitor import train, test

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--test", help="Show pretrained agent in environment", action="store_true")
parser.add_argument("--no_graphics", help="Do not show graphics during training", action="store_true")
parser.add_argument("--environment", nargs='?', help="Pick environment file", default="env/Banana.exe")
parser.add_argument("--checkpoint", nargs='?', help="Pick environment file", default="logs/checkpoint.pth")
parser.add_argument("--model_name", nargs='?', help="Choose a model name. Options: DQN, DuelDQN", default="DQN")
parser.add_argument("--double", help="Enable double DQN", action="store_true")

if __name__ == '__main__':

    # Get arguments
    args = parser.parse_args()    

    # Setup agent
    agent = Agent(state_size=37, action_size=4, model_name=args.model_name, enable_double=args.double, random_state=42)

    # Testing or training
    if args.test:

        # Get environment (with graphics)
        env = UnityEnvironment(file_name=args.environment, seed=42)
        test(env, agent, brain_name=env.brain_names[0], checkpoint=args.checkpoint)
    else:

        # Get environment (no graphics)
        env = UnityEnvironment(file_name=args.environment, seed=42, no_graphics=args.no_graphics)
        train(env, agent, 
            brain_name=env.brain_names[0], 
            episodes=1000,
            eps_start=1.0, 
            eps_end=0.001, 
            eps_decay=0.97, 
            thr_score=13.0
        )

    # Close environment when done
    env.close()
