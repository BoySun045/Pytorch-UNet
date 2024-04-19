import numpy as np
import argparse

# read npy file from a path and print shape and min max values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path to npy file')
    args = parser.parse_args()
    path = args.path

    arr = np.load(path)
    arr = arr['weights']
    print(f"shape: {arr.shape}")
    print(f"min: {arr.min()}")
    print(f"max: {arr.max()}")

if __name__ == '__main__':
    main()

