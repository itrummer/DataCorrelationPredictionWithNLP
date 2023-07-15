# compare performance for different metrics
from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs
)
from sklearn.model_selection import train_test_split
from spellchecker import SpellChecker
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import os
import random as rand
import sys
import time
import wandb

# get command line parameters
coeff = sys.argv[1]
min_v1 = float(sys.argv[2])
max_v2 = float(sys.argv[3])
mod_type = sys.argv[4]
mod_name = sys.argv[5]
scenario = sys.argv[6]
test_ratio = float(sys.argv[7])
print(f'Coefficients: {coeff}')
print(f'Minimal value 1: {min_v1}')
print(f'Maximal value 2: {max_v2}')
print(f'Model type: {mod_type}')
print(f'Model name: {mod_name}')
print(f'Scenario: {scenario}')
print(f'Test ratio: {test_ratio}')

# initialize for deterministic results
seed = 42
rand.seed(seed)

# load data
path = 'data/corresult4.csv'
data = pd.read_csv(path, sep = ',')
data = data.sample(frac=1, random_state=seed)
data.columns = ['dataid', 'datapath', 'nrrows', 'nrvals1', 'nrvals2', 
                'type1', 'type2', 'column1', 'column2', 'method',
                'coefficient', 'pvalue', 'time']

# filter data
data = data[data['method']==coeff]
nr_total = len(data.index)
print(f'Nr. samples: {nr_total}')
print('Sample from filtered data:')
print(data.head())

# |coefficient|>=min_val1 and |pvalue|<=max_val2  -> label 1
def coefficient_label(row):
  if abs(row['coefficient']) >= min_v1 and abs(row['pvalue']) <= max_v2:
    return 1
  else:
    return 0
data['label'] = data.apply(coefficient_label, axis=1)

# split data into training and test set
def def_split(data):
  print('Data sets in training and test may overlap')
  x_train, x_test, y_train, y_test = train_test_split(
      data[['column1', 'column2']], data['label'],
      test_size=test_ratio, random_state=seed)
  train = pd.concat([x_train, y_train], axis=1)
  test = pd.concat([x_test, y_test], axis=1)
  print(f'train shape: {train.shape}')
  print(f'test shape: {test.shape}')
  return train, test

# split samples based on their data set
def ds_split(data):
  print('Separating training and test sets by data')
  counts = data['dataid'].value_counts()
  print(f'Counts: {counts}')
  print(f'Count.index: {counts.index}')
  print(f'Count.index.values: {counts.index.values}')
  print(f'counts.shape: {counts.shape}')
  print(f'counts.iloc[0]: {counts.iloc[0]}')
  nr_vals = len(counts)
  nr_test_ds = int(nr_vals * test_ratio)
  print(f'Nr. test data sets: {nr_test_ds}')
  ds_ids = counts.index.values.tolist()
  print(type(ds_ids))
  print(ds_ids)
  test_ds = rand.sample(ds_ids, nr_test_ds)
  print(f'TestDS: {test_ds}')
  def is_test(row):
    if row['dataid'] in test_ds:
      return True
    else:
      return False
  data['istest'] = data.apply(is_test, axis=1)
  train = data[data['istest'] == False]
  test = data[data['istest'] == True]
  print(f'train.shape: {train.shape}')
  print(f'test.shape: {test.shape}')
  print(train)
  print(test)
  return train[['column1', 'column2', 'label']], test[['column1', 'column2', 'label']]

# split into test and training
if scenario == 'defsep':
    train, test = def_split(data)
elif scenario == 'datasep':
    train, test = ds_split(data)
else:
    print(f'Error - undefined scenario!')
    sys.exit(1)
train.columns = ['text_a', 'text_b', 'labels']
test.columns = ['text_a', 'text_b', 'labels']
print(train.head())
print(test.head())

# prepare loss scaling
lab_counts = train['labels'].value_counts()
nr_zeros = lab_counts.loc[0]
nr_ones = lab_counts.loc[1]
nr_all = float(len(train.index))
weights = [nr_all/nr_zeros, nr_all/nr_ones]

# train classification model
w_name = f'{coeff};{min_v1};{max_v2};{mod_type};{mod_name};{scenario}'
s_time = time.time()
model_args = ClassificationArgs(num_train_epochs=10, train_batch_size=100, 
                                eval_batch_size=100,
                                overwrite_output_dir=True, manual_seed=seed,
                                evaluate_during_training=True, no_save=True,
                                wandb_project='CorrelationPredictionv1',
                                wandb_kwargs={'name': w_name})
model = ClassificationModel(mod_type, mod_name, weight=weights,
                            use_cuda = True, args=model_args)
model.train_model(train_df=train, eval_df=test, acc=metrics.accuracy_score, 
    rec=metrics.recall_score, pre=metrics.precision_score, f1=metrics.f1_score)
training_time = time.time() - s_time

# a simple baseline determining correlation based on Jaccard similarity
def baseline(col_pairs):
    predictions = []
    for cp in col_pairs:
        c1 = cp[0]
        c2= cp[1]
        s1 = set(c1.split())
        s2 = set(c2.split())
        ns1 = len(s1)
        ns2 = len(s2)
        ni = len(set.intersection(s1, s2))
        # calculate Jaccard coefficient
        jac = ni / (ns1 + ns2 - ni)
        # predict correlation if similar
        if jac > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

# log all metrics into summary for data subset
def log_metrics(sub_test, test_name, lb, ub, pred_method):
    sub_test.columns = ['text_a', 'text_b', 'labels', 
            'length', 'nrtokens', 'wordratio']
    # print out a sample for later analysis
    print(f'Sample for test {test_name}:')
    sample = sub_test.sample(frac=0.1)
    print(sample)
    # predict correlation via baseline or model
    sub_test = sub_test[['text_a', 'text_b', 'labels']]
    samples = []
    for ri, r in sub_test.iterrows():
        samples.append([r['text_a'], r['text_b']])
    s_time = time.time()
    if pred_method == 0:
        preds = baseline(samples)
    else:
        preds = model.predict(samples)[0]
    # log various performance metrics
    t_time = time.time() - s_time
    nr_samples = len(sub_test.index)
    t_per_s = float(t_time) / nr_samples
    f1 = metrics.f1_score(sub_test['labels'], preds)
    pre = metrics.precision_score(sub_test['labels'], preds)
    rec = metrics.recall_score(sub_test['labels'], preds)
    acc = metrics.accuracy_score(sub_test['labels'], preds)
    mcc = metrics.matthews_corrcoef(sub_test['labels'], preds)
    f1_name = f'{test_name}f1'
    pre_name = f'{test_name}pre'
    rec_name = f'{test_name}rec'
    acc_name = f'{test_name}acc'
    mcc_name = f'{test_name}mcc'
    tps_name = f'{test_name}tps'
    lb_name = f'{test_name}lb'
    ub_name = f'{test_name}ub'
    wandb.log({f1_name : f1, pre_name : pre, rec_name : rec, 
        acc_name : acc, mcc_name : mcc, tps_name : t_per_s,
        lb_name : lb, ub_name : ub})
    # also log to local file
    with open('results.csv', 'a+') as file:
        file.write(f'{coeff},{min_v1},{max_v2},"{mod_type}",' \
                f'"{mod_name}","{scenario}",{test_ratio},' \
                f'"{test_name}",{pred_method},{lb},{ub},' \
                f'{f1},{pre},{rec},{acc},{mcc},{t_per_s},' \
                f'{training_time}\n')

# test dependency on column name properties
def names_length(row):
    return len(row['text_a']) + len(row['text_b'])
test['length'] = test.apply(names_length, axis=1)
def names_tokens(row):
    return row['text_a'].count(' ') + row['text_b'].count(' ')
test['nrtokens'] = test.apply(names_tokens, axis=1)
spell = SpellChecker(distance=0)
def word_ratio(row):
    col_text = row['text_a'] + ' ' + row['text_b']
    tokens = col_text.split()
    nr_tokens = len(tokens)
    nr_words = len(spell.known(tokens))
    return float(nr_words) / nr_tokens
test['wordratio'] = test.apply(word_ratio, axis=1)

# use simple baseline and model for prediction
for m in [0, 1]:
    # use entire test set (redundant - for verification)
    test_name = f'{m}-final'
    log_metrics(test, test_name, 0, 'inf', m)
    
    # test for data types
    for type1 in ['object', 'float64', 'int64', 'bool']:
        for type2 in ['object', 'float64', 'int64', 'bool']:
            sub_test = test.query(f'type1=={type1} and type2=={type2}')
            test_name = f'Types{type1}-{type2}'
            log_metrics(sub_test, test_name, -1, -1, m)
    
    # test for different subsets
    for q in [(0, 0.25), (0.25, 0.5), (0.5, 1)]:
        qlb = q[0]
        qub = q[1]
        # column name length
        lb = test['length'].quantile(qlb)
        ub = test['length'].quantile(qub)
        sub_test = test[(test['length'] >= lb) & (test['length'] <= ub)]
        test_name = f'L{m}-{qlb}-{qub}'
        log_metrics(sub_test, test_name, lb, ub, m)
        # number of tokens in column names
        lb = test['nrtokens'].quantile(qlb)
        ub = test['nrtokens'].quantile(qub)
        sub_test = test[(test['nrtokens'] >= lb) & (test['nrtokens'] <= ub)]
        test_name = f'N{m}-{qlb}-{qub}'
        log_metrics(sub_test, test_name, lb, ub, m)
        # ratio of English words in column names
        lb = test['wordratio'].quantile(qlb)
        ub = test['wordratio'].quantile(qub)
        sub_test = test[(test['wordratio'] >= lb) & (test['wordratio'] <= ub)]
        test_name = f'W{m}-{qlb}-{qub}'
        log_metrics(sub_test, test_name, lb, ub, m)
