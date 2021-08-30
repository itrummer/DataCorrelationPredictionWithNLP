'''
Created on Aug 27, 2021

@author: immanueltrummer
'''
import pandas as pd
import sklearn.metrics
import statistics

in_dir = '/tmp/neat/'
for in_path in [
    f'{in_dir}/pearson.csv', 
    f'{in_dir}/spearman.csv']:
    
    df = pd.read_csv(in_path, sep=',')
    labels = df['labels']
    p_mean = df['predictions'].mean()
    p_med = statistics.median(df['predictions'])
    b_preds = df['predictions'].apply(lambda x:1 if x>p_med else 0)
    
    rec = sklearn.metrics.recall_score(y_true=labels, y_pred=b_preds)
    pre = sklearn.metrics.precision_score(y_true=labels, y_pred=b_preds)
    f1 = sklearn.metrics.f1_score(y_true=labels, y_pred=b_preds)
    print(f'R {rec} P {pre} F1 {f1}')