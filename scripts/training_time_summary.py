"""Mean and standard deviation of each timing column written by training_time.py."""
import pandas


def main(args):
    # Read the CSV file
    df = pandas.read_csv(args.input)
    with open(args.output, 'w') as f:
        for column in df.columns[1:]:
            f.write(column+': '+str(df[column].mean())+" +- "+str(df[column].std())+'\n')
            print(column+': '+str(df[column].mean())+" +- "+str(df[column].std()))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input CSV file', default='../outputs/training_time_0.csv')
    parser.add_argument('--output', help='Output text file', default='../outputs/training_time_means_0.txt')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
