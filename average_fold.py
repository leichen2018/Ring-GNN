from utils import average_csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output-file', type=str, default='output')
parser.add_argument('--output-folder', type=str, default='results')
args = parser.parse_args()

average_csv(args.output_folder, args.output_file)