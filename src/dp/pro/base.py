'''
Created on Aug 23, 2021

@author: immanueltrummer
'''
import argparse
import pandas as pd


def write_stats(df, out_path):
    """ Write statistics to output file.
    
    Args:
        df: data frame with method-specific row order
        out_path: write statistics into this file
    """
    # cost = df['time'].cumsum()
    df['hits'] = df['labels'].cumsum()
    # stats = pd.DataFrame([cost, hits])
    # stats.columns = ['cost', 'hits']
    df = df.reset_index()
    df.index.name = 'step'
    df = df.loc[::100,['hits', 'column1', 'column2']]
    df.to_csv(out_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Path to predictions file')
    parser.add_argument('out_pre', type=str, help='Path prefix of output files')
    args = parser.parse_args()
    
    df = pd.read_csv(args.in_path, sep=',')
    print(df.info())
    write_stats(df, f'{args.out_pre}bydata.csv')
    
    df.sort_values(
        axis=0, ascending=False, 
        inplace=True, by='predictions')
    write_stats(df, f'{args.out_pre}simple.csv')
    
    df = df.sample(frac=1)
    write_stats(df, f'{args.out_pre}random.csv')
    
    def priority(row):
        pass
    df['priority'] = df.apply(lambda r:priority(r), axis='columns')
    df.sort_values(
        axis=0, ascending=False, 
        inplace=True, by='priority')
    write_stats(df, f'{args.out_pre}priority.csv')