class Env_Args:
    def __init__(self):
        self.map_file = "map.txt"
        self.n_robots = 5
        self.n_packages = 20
        self.episode_limit = 100
        self.move_cost = -0.01
        self.delivery_reward = 10.0
        self.delay_reward = 1.0