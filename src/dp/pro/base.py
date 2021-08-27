'''
Created on Aug 23, 2021

@author: immanueltrummer
'''
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer, util


def write_stats(df, out_path, scale, pred_time):
    """ Write statistics to output file.
    
    Args:
        df: data frame with method-specific row order
        out_path: write statistics into this file
        scale: simulate scaling data by this factor
        pred_time: time required for predictions
    """
    df['crows'] = df['nrrows'].cumsum()
    df['ctime'] = df['time'].cumsum().mul(scale)
    df['ctime'] = df['ctime'].add(pred_time)
    df['chits'] = df['labels'].cumsum()
    df = df.reset_index()
    df.index.name = 'step'
    df = df.loc[::100,['crows', 'ctime', 'chits', 'column1', 'column2']]
    df.to_csv(out_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Path to predictions file')
    parser.add_argument('out_pre', type=str, help='Path prefix of output files')
    args = parser.parse_args()
    
    df = pd.read_csv(args.in_path, sep=',')
    print(df.info())
    
    for scale in [1, 10, 100, 1000]:
        prefix = f'{args.out_pre}_F{scale}_'
        write_stats(df, f'{prefix}bydata.csv', scale, 0)
        
        df.sort_values(
            axis=0, ascending=False, 
            inplace=True, by='predictions')
        write_stats(df, f'{prefix}simple.csv', scale, 26)
        
        df = df.sample(frac=1)
        write_stats(df, f'{prefix}random.csv', scale, 0)
        
        # def priority(row):
            # return row['predictions'] * (2 if len(str(row['column1'])) > 50 else 1)
        # df['priority'] = df.apply(lambda r:priority(r), axis='columns')    
        # df.sort_values(
            # axis=0, ascending=False, 
            # inplace=True, by='priority')
        # write_stats(df, f'{prefix}priority.csv', scale, 26)
    
    # def similarity(row):
        # embedding1 = row['embedding1']
        # embedding2 = row['embedding2']
        # return util.pytorch_cos_sim(embedding1, embedding2)
    # df['similarity'] = df.apply(lambda r:similarity(r), axis='columns')
    # df.sort_values(axis=0, ascending=False, inplace=True, by='similarity')
    # write_stats(df, f'{args.out_pre}similarity.csv')