from runners import Runner_MAPPO, Runner_QMIX
from config import MAPPO_Args, Env_Args, QMIX_Args
import argparse


def main():
    parser = argparse.ArgumentParser("MAPPO or QMIX")
    parser.add_argument("--algorithm", type=str, default="MAPPO")
    MorQ = parser.parse_args()
    if MorQ.algorithm == "MAPPO":
        args = MAPPO_Args()
        env_args = Env_Args()

        runner = Runner_MAPPO(args=args, env_args=env_args, seed=2025)
        runner.run()
    else:
        args = QMIX_Args()
        env_args = Env_Args()
        
        runner = Runner_QMIX(args=args, env_args=env_args, seed=2025)
        runner.run()

if __name__ == "__main__":
    main()