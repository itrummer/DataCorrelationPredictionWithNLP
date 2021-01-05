# compare performance for different metrics
from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs
)
from sklearn.model_selection import train_test_split
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
path = '../data/corresult4.csv'
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
model_args = ClassificationArgs(num_train_epochs=1, train_batch_size=100,
                                overwrite_output_dir=True, manual_seed=seed,
                                evaluate_during_training=True, no_save=True,
                                wandb_project='CorrelationPredictionv1',
                                wandb_kwargs={'name': w_name})
model = ClassificationModel(mod_type, mod_name, weight=weights,
                            use_cuda = True, args=model_args)
model.train_model(train_df=train, eval_df=test, acc=metrics.accuracy_score, 
    rec=metrics.recall_score, pre=metrics.precision_score, f1=metrics.f1_score)

# log all metrics into summary for data subset
def log_metrics(sub_test, test_name):
    sub_test.columns = ['text_a', 'text_b', 'labels', 'length', 'nrtokens']
    sub_test = sub_test[['text_a', 'text_b', 'labels']]
    samples = []
    for ri, r in sub_test.iterrows():
        samples.append([r['text_a'], r['text_b']])
    s_time = time.time()
    preds = model.predict(samples)[0]
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
    wandb.log({f1_name : f1, pre_name : pre, rec_name : rec, 
        acc_name : acc, mcc_name : mcc, tps_name : t_per_s})
    #wandb.run.summary[test_name + 'f1'] = f1
    #wandb.run.summary[test_name + 'pre'] = pre
    #wandb.run.summary[test_name + 'rec'] = rec
    #wandb.run.summary[test_name + 'acc'] = acc
    #wandb.run.summary[test_name + 'mcc'] = mcc
    #wandb.run.summary[test_name + 'tps'] = t_per_s
    #wandb.run.summary.update()

# test dependency on column name properties
def names_length(row):
    return len(row['text_a']) + len(row['text_b'])
test['length'] = test.apply(names_length, axis=1)
def names_tokens(row):
    return row['text_a'].count(' ') + row['text_b'].count(' ')
test['nrtokens'] = test.apply(names_tokens, axis=1)

for q in [(0, 0.25), (0.25, 0.5), (0.5, 1)]:
    qlb = q[0]
    qub = q[1]
    # column name length
    lb = test['length'].quantile(qlb)
    ub = test['length'].quantile(qub)
    sub_test = test[(test['length'] >= lb) & (test['length'] <= ub)]
    test_name = f'L{lb}-{ub}'
    log_metrics(sub_test, test_name)
    # data set size
    lb = test['nrtokens'].quantile(qlb)
    ub = test['nrtokens'].quantile(qub)
    sub_test = test[(test['nrtokens'] >= lb) & (test['nrtokens'] <= ub)]
    test_name = f'N{lb}-{ub}'
    log_metrics(sub_test, test_name)

# test dependency on column types
#ptypes = ['int64', 'float64', 'object', 'bool', 'datetime64', 'timedelta[ns]', 'category']
#for c1_type in ptypes:
#    for c2_type in ptypes:
#        sub_test = test[(test['type1'] == c1_type) & (test['type2'] == c2_type)]
#        test_name = f'T{c1_type}-{c2_type}'
#        log_metrics(sub_test, test_name)
