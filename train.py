from runners.mappo_runner import Runner_MAPPO
from config import MAPPO_Args, Env_Args


def main():
    args = MAPPO_Args()
    env_args = Env_Args()

    runner = Runner_MAPPO(args=args, env_args=env_args, seed=2025)
    runner.run()

if __name__ == "__main__":
    main()