'''
Created on Aug 24, 2021

@author: immanueltrummer
'''
import argparse
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str, help='Path to input directory')
    parser.add_argument('out_dir', type=str, help='Path to output directory')
    args = parser.parse_args()
    
    for scale in [1, 10, 100, 1000]:
        _, axes = plt.subplots(nrows=2, ncols=1, figsize=(3.5,2.5))
        # for file_name in [
            # 'alltablesbydata.csv', 'alltablespriority.csv', 'alltablesrandom.csv', 
            # 'alltablessimple.csv', 'alltablessimilarity.csv', 'alltablesrl.csv']:
        for file_name in [f'alltables_F{scale}_rl.csv', 
                          f'alltables_F{scale}_random.csv', 
                          f'alltables_F{scale}_simple.csv']:
            df = pd.read_csv(f'{args.in_dir}/{file_name}')
            hits = df.loc[:,'chits']
            step = df.loc[:,'step']
            cost = df.loc[:,'crows']
            time = df.loc[:,'ctime']
            #axes[0].plot(step, hits)
            axes[1].plot(time, hits)
            axes[1].set_xscale('log')
        axes[1].legend(['RL', 'Random', 'Predictions'])
        plt.savefig(f'{args.out_dir}/F{scale}.pdf')