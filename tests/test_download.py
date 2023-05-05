import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwuhub.datasets import Procgen

if __name__ == "__main__":
    procgen = Procgen()
    print(procgen.load_scores()['PPO'].shape)