from argparse import ArgumentParser


def run():
    pass


if __name__ == '__main__':
    parser = ArgumentParser(description="basic paser for bandit problem")
    parser.add_argument('--datapath', type=str, default='data/heart.dat')
    parser.add_argument('--algo', type=str, default='LMC')

    run()
    print('Done!')
