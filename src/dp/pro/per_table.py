'''
Created on Aug 29, 2021

@author: immanueltrummer
'''
import argparse
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pandas as pd
import statistics

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str, help='Path to input directory')
    parser.add_argument('out_dir', type=str, help='Path to output directory')
    args = parser.parse_args()
    
    # plt.rcParams.update(
        # {'text.usetex': True, 'font.size':9,  
         # 'font.serif':['Computer Modern'],
         # 'font.family':'serif'})
    b_ratios = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    for coefficient in ['pearson', 'spearman', 'theilsu']:
        print(f'Processing {coefficient} correlation ...')
        in_path = f'{args.in_dir}/{coefficient}.csv'
        df = pd.read_csv(in_path, sep=',')

        # _, axes = plt.subplots(nrows=3, ncols=1, figsize=(3.5,5),
            # subplotpars=matplotlib.figure.SubplotParams(
                # wspace=0.425, hspace=0.75))
        
        for plot_idx, min_pairs in enumerate([0, 10, 40]):
            print(f'Processing tables with at least {min_pairs} columns ...')
            r_line = []
            j_line = []
            n_line = []
            
            # Analyze correlation between predictors and inter-column correlation
            for predictor_col in ['jaccard', 'predictions']:
                cor_samples = []
                nr_useful = 0
                for _, g in df.groupby('dataid'):
                    if g.shape[0] >= min_pairs:
                        cor_with = g.coefficient
                        cor_sample = g[predictor_col].corr(cor_with, method='spearman')
                        if not math.isnan(cor_sample):
                            cor_samples.append(cor_sample)
                            nr_useful += 1
                        # else:
                            # cor_samples.append(0)
                
                cor_mean = statistics.mean(cor_samples)
                print(
                    f'{coefficient}; Predictor: {predictor_col}; '
                    f'Correlation: {cor_mean}; Used: {nr_useful}'
                    )
            
            for x in b_ratios:
                r_data = []
                j_data = []
                n_data = []
                for _, g in df.groupby('dataid'):
                    if g.shape[0] >= min_pairs:
                        nr_pairs = g.shape[0]
                        nr_cors = g['labels'].sum()
                        if nr_cors > 0:
                            budget = round(nr_pairs * x)
                            
                            g.sort_values(
                                axis=0, ascending=False, inplace=True, 
                                ignore_index=True, by='predictions')
                            nr_hits = g.loc[0:budget,'labels'].sum()
                            n_data.append(nr_hits/nr_cors)
                            
                            g.sort_values(
                                axis=0, ascending=False, inplace=True,
                                ignore_index=True, by='jaccard')
                            nr_hits = g.loc[0:budget,'labels'].sum()
                            j_data.append(nr_hits/nr_cors)
                            
                            g = g.sample(frac=1, ignore_index=True)
                            nr_hits = g.loc[0:budget,'labels'].sum()
                            r_data.append(nr_hits/nr_cors)
                        
                r_line.append(statistics.mean(r_data))
                j_line.append(statistics.mean(j_data))
                n_line.append(statistics.mean(n_data))

            # random_result = df.groupby('dataid').apply(
                # lambda g:g.reset_index().loc[
                    # 0:round(g.shape[0]/10.0),'labels'].sum()/max(1,g['labels'].sum()))
            # nlp_result = df.groupby('dataid').apply(
                # lambda g:g.sort_values(
                    # axis=0, ascending=False, 
                    # by='predictions').reset_index().loc[
                        # 0:round(g.shape[0]/10.0),'labels'].sum()/max(1,g['labels'].sum()))


            # print(f'Random: {r_line}')
            # print(f'Jaccard: {j_line}')
            # print(f'NEAT: {n_line}')
            
            # cur_axis = axes[plot_idx]
            # cur_axis.plot(b_ratios, r_line, 'b1-')
            # cur_axis.plot(b_ratios, n_line, 'rx-')
            # cur_axis.yaxis.grid()
            # if min_pairs == 0:
                # cur_axis.set_title(f'All Tables')
            # else:
                # cur_axis.set_title(f'At Least {min_pairs} Column Pairs')
            # cur_axis.set_ylabel('Detections')
            # # if plot_idx == 3:
            # cur_axis.set_xlabel('Number of Tests')
            # cur_axis.legend(['Random', '+NLP'])
            #
        # plt.tight_layout(1.05)
        # plt.savefig(f'{args.out_dir}/{coefficient}.pdf')