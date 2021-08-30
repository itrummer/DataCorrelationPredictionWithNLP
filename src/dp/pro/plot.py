'''
Created on Aug 24, 2021

@author: immanueltrummer
'''
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str, help='Path to input directory')
    parser.add_argument('out_dir', type=str, help='Path to output directory')
    args = parser.parse_args()
    
    plt.rcParams.update({'text.usetex': True, 'font.size':9})
    
    for coefficient in ['pearson', 'spearman', 'theilsu']:
        _, axes = plt.subplots(nrows=4, ncols=1, figsize=(3.5,6),
            subplotpars=matplotlib.figure.SubplotParams(
                wspace=0.425, hspace=0.75))
        for plot_idx, scale in enumerate([1, 10, 100, 1000]):
            for file_name, line_style in [
                #(f'alltables_F{scale}_rl.csv', 'b1-'),
                (f'alltables_F{scale}_random.csv', 'rx-'),
                (f'alltables_F{scale}_byrows.csv', 'b1-'),
                (f'alltables_F{scale}_simple.csv', 'g.-'),
                (f'alltables_F{scale}_rowpred.csv', 'y,-')]:
                df = pd.read_csv(f'{args.in_dir}/{coefficient}/{file_name}')
                hits = df.loc[:,'chits']
                time = df.loc[:,'ctime']
                cur_axis = axes[plot_idx]
                cur_axis.plot(time, hits, line_style, markersize=5)
                cur_axis.set_xscale('linear')
                cur_axis.set_yscale('linear')
                cur_axis.set_title(f'Scaling Factor: {scale}')
                cur_axis.set_ylabel('Detections')
                cur_axis.legend(
                    #['RL', 'Random', 'Predict'], 
                    ['Random', 'Size', 'Predict', 'Pred+Size'],
                    ncol=1)
                if plot_idx == 3:
                    cur_axis.set_xlabel('Time (s)')
            cur_axis.yaxis.grid()
    
        plt.tight_layout(1.05)
        plt.savefig(f'{args.out_dir}/{coefficient}/time_vs_hits.pdf')