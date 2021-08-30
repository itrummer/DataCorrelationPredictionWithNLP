'''
Created on Aug 23, 2021

@author: immanueltrummer
'''
import argparse
import math
import pandas as pd

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
    # df['logrows'] = df.apply(
        # lambda r:round(math.log10(max(1,r['nrrows']))), 
        # axis='columns')
    df['whits'] = df['labels'] * df['nrrows']
    df['chits'] = df['whits'].cumsum()
    df = df.reset_index()
    df.index.name = 'step'
    df = df.loc[:,['crows', 'ctime', 'chits', 'column1', 'column2']]
    df.to_csv(out_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str, help='Path to input directory')
    parser.add_argument('out_dir', type=str, help='Path to output directory')
    args = parser.parse_args()
    
    inference_times = {'pearson':39, 'spearman':54, 'theilsu':118}
    for coefficient in ['pearson', 'spearman', 'theilsu']:
        in_path = f'{args.in_dir}/{coefficient}.csv'
        df = pd.read_csv(in_path, sep=',')
        inference_time = inference_times[coefficient]
        for scale in [1, 10, 100, 1000]:
            prefix = f'{args.out_dir}/{coefficient}/alltables_F{scale}_'
            write_stats(df, f'{prefix}bydata.csv', scale, 0)
            
            df.sort_values(
                axis=0, ascending=False, 
                inplace=True, by='predictions')
            write_stats(df, f'{prefix}simple.csv', scale, inference_time)
            
            df.sort_values(axis=0, ascending=True, inplace=True, by='nrrows')
            write_stats(df, f'{prefix}byrows.csv', scale, 0)
            
            df = df.sample(frac=1)
            write_stats(df, f'{prefix}random.csv', scale, 0)
            
            def row_pred_val(r):
                """ Calculates priority based on row size and predictions. """
                raw_nr_rows = r['nrrows']
                # if raw_nr_rows * scale < 10000:
                    # return float('inf')
                # else:
                pos_nr_rows = max(1, raw_nr_rows)
                return r['predictions'] * pos_nr_rows
                    # log_size = round(math.log10(scale * pos_nr_rows))
                    # row_pred = log_size - r['predictions']
                    # return row_pred

            df['rowpred'] = df.apply(row_pred_val, axis='columns')
            df.sort_values(axis=0, ascending=False, inplace=True, by='rowpred')
            write_stats(df, f'{prefix}rowpred.csv', scale, inference_time/3)
        
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