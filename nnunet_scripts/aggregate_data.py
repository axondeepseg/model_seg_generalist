import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Where the datasets are located")
    parser.add_argument("-o", "--output", help="Where to save the aggregated dataset")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

if __name__ == "__main__":
    main()
