from runners.mappo import Runner_MAPPO


def main():
    runner = Runner_MAPPO("map.txt")
    runner.run()

if __name__ == "__main__":
    main()