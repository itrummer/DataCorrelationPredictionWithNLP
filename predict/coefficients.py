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
import wandb

# initialize for deterministic results
seed = 0
rand.seed(seed)

# load data
path = '../data/corresult4.csv'
data = pd.read_csv(path, sep = ',')
data = data.sample(frac=1, random_state=seed)
data.columns = ['dataid', 'datapath', 'nrrows', 'nrvals1', 'nrvals2', 
                'type1', 'type2', 'column1', 'column2', 'method',
                'coefficient', 'pvalue', 'time']

# divide data into subsets
pearson = data[data['method']=='pearson']
spearman = data[data['method']=='spearman']
theilsu = data[data['method']=='theilsu']
print('Pearson data:')
print(pearson.head())
print('Spearman data:')
print(spearman.head())
print('Theil\'s u data:')
print(theilsu.head())

# generate and print data statistics
nr_ps = len(pearson.index)
nr_sm = len(spearman.index)
nr_tu = len(theilsu.index)
print(f'#Samples for Pearson: {nr_ps}')
print(f'#Samples for Spearman: {nr_sm}')
print(f'#Samples for Theil\'s u: {nr_tu}')

# |coefficient>0.5| -> label 1
def coefficient_label(row):
  if abs(row['coefficient']) > 0.5:
    return 1
  else:
    return 0
pearson['label'] = pearson.apply(coefficient_label, axis=1)
spearman['label'] = spearman.apply(coefficient_label, axis=1)
theilsu['label'] = theilsu.apply(coefficient_label, axis=1)

rc_p = len(pearson[pearson['label']==1].index)/nr_ps
rc_s = len(spearman[spearman['label']==1].index)/nr_sm
rc_u = len(theilsu[theilsu['label']==1].index)/nr_tu
print(f'Ratio correlated - Pearson: {rc_p}')
print(f'Ratio correlated - Spearman: {rc_s}')
print(f'Ratio correlated - Theil\s u: {rc_u}')

# split data into training and test set
def def_split(data):
  x_train, x_test, y_train, y_test = train_test_split(
      pearson[['column1', 'column2']], pearson['label'],
      test_size=0.2, random_state=seed)
  train = pd.concat([x_train, y_train], axis=1)
  test = pd.concat([x_test, y_test], axis=1)
  return train, test

# split samples based on their data set
def ds_split(data):
  counts = data['dataid'].value_counts()
  print(f'Counts: {counts}')
  print(f'Count.index: {counts.index}')
  print(f'Count.index.values: {counts.index.values}')
  print(f'counts.shape: {counts.shape}')
  print(f'counts.iloc[0]: {counts.iloc[0]}')
  nr_vals = len(counts)
  nr_test_ds = int(nr_vals * 0.2)
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

# iterate over different coefficients
for c_data in [pearson, spearman, theilsu]:
    # configure WandDB
    c_name = c_data.iloc[0,9]
    print(f'coefficients: {c_name}')
    #os.environ['WANDB_NAME'] = f'coefficient {c_name}'
    #os.environ['WANDB_NOTES'] = 'test performance of Roberta with test ratio 0.2 on different coefficients'
    #cur_run = wandb.init(reinit=True)
    # split into test and training
    train, test = ds_split(c_data)
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
    model_args = ClassificationArgs(num_train_epochs=10, train_batch_size=100,
                                    overwrite_output_dir=True, manual_seed=seed,
                                    evaluate_during_training=True, no_save=True,
                                    wandb_project='CorrelationPredictionv1',
                                    wandb_kwargs={'name': c_name})
    model = ClassificationModel("roberta", "roberta-base", weight=weights,
                                use_cuda = True, args=model_args)
    model.train_model(train_df=train, eval_df=test, acc=metrics.accuracy_score, 
        rec=metrics.recall_score, pre=metrics.precision_score, f1=metrics.f1_score)
    # inform Wandb that current run ends
    #cur_run.finish()
