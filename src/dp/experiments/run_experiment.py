'''
Created on Aug 12, 2023

@author: immanueltrummer
'''
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

import argparse
import sklearn.metrics as metrics
import pandas as pd
import random as rand
import time

from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs
)
from sklearn.model_selection import train_test_split


def add_type(row):
    """ Enrich column name by adding column type.
    
    Args:
        row: describes correlation between two columns.
    
    Returns:
        row with enriched column names.
    """
    row['column1'] = row['column1'] + ' ' + row['type1']
    row['column2'] = row['column2'] + ' ' + row['type2']
    return row


def def_split(data, test_ratio, seed):
    """ Split data into training and test set.
    
    With this approach, different column pairs from the
    same data set may appear in training and test set.
    
    Args:
        data: a pandas dataframe containing all data.
        test_ratio: ratio of test cases after split.
        seed: random seed for deterministic results.
    
    Returns:
        a tuple containing training, then test data.
    """
    print('Data sets in training and test may overlap')
    x_train, x_test, y_train, y_test = train_test_split(
      data[['column1', 'column2', 'type1', 'type2']], data['label'],
      test_size=test_ratio, random_state=seed)
    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)
    print(f'train shape: {train.shape}')
    print(f'test shape: {test.shape}')
    return train, test


def ds_split(data, test_ratio):
    """ Split column pairs into training and test samples.
    
    With this method, training and test set contain columns
    of disjunct data sets, making prediction a bit harder.
    
    Args:
        data: a pandas dataframe containing all data.
        test_ratio: ratio of test cases after splitting.
    
    Returns:
        a tuple containing training, then test set.
    """
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
    return train[
        ['column1', 'column2', 'type1', 'type2', 'label']], test[
            ['column1', 'column2', 'type1', 'type2', 'label']]


def baseline(col_pairs):
    """ A simple baseline predicting correlation via Jaccard similarity.
    
    Args:
        col_pairs: list of tuples with column names.
    
    Returns:
        list of predictions (1 for correlation, 0 for no correlation).
    """
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
def log_metrics(
        coeff, min_v1, max_v2, mod_type, mod_name, scenario, 
        test_ratio, sub_test, test_name, lb, ub, pred_method,
        out_path):
    """ Predicts using baseline or model, writes metrics to file.
    
    Args:
        coeff: predict correlation according to this coefficient.
        min_v1: lower bound on coefficient value for correlation.
        max_v2: upper bound on p-value to be considered correlated.
        mod_type: base type of language model used for prediction.
        mod_name: precise name of language model used for prediction.
        scenario: how training and test data relate to each other.
        test_ratio: ratio of column pairs used for testing (not training).
        sub_test: data frame with test cases, possibly a subset.
        test_name: write this test name into result file.
        lb: lower bound on a test-specific metric constraining test cases.
        ub: upper bound on test-specific metric, constraining test cases.
        pred_metho: whether to use language model or simple baseline.
        out_path: path to result output file (results are appended).
    """
    sub_test.columns = [
        'text_a', 'text_b', 'type1', 'type2', 'labels', 'length', 'nrtokens']
    # print out a sample for later analysis
    print(f'Sample for test {test_name}:')
    sample = sub_test.sample(frac=0.1)
    print(sample)
    # predict correlation via baseline or model
    sub_test = sub_test[['text_a', 'text_b', 'labels']]
    samples = []
    for _, r in sub_test.iterrows():
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
    # also log to local file
    with open(out_path, 'a+') as file:
        file.write(f'{coeff},{min_v1},{max_v2},"{mod_type}",' \
                f'"{mod_name}","{scenario}",{test_ratio},' \
                f'"{test_name}",{pred_method},{lb},{ub},' \
                f'{f1},{pre},{rec},{acc},{mcc},{t_per_s},' \
                f'{training_time}\n')


def names_length(row):
    """ Calculate combined length of column names.
    
    Args:
        row: contains information on one column pair.
    
    Returns:
        combined length of column names (in characters).
    """
    return len(row['text_a']) + len(row['text_b'])

def names_tokens(row):
    """ Calculates number of tokens (separated by spaces).
    
    Attention: this is not the number of tokens as calculated
    by the tokenizer of the language model but an approximation.
    
    Args:
        row: contains information on one column pair.
    
    Returns:
        number of space-separated substrings in both column names.
    """
    return row['text_a'].count(' ') + row['text_b'].count(' ')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('src_path', type=str, help='Path to source file')
    parser.add_argument(
        'coeff', type=str, help='Correlation coefficient (e.g., "pearson")')
    parser.add_argument(
        'min_v1', type=float, 
        help='Minimal coefficient value for correlation (e.g., 0.9)')
    parser.add_argument(
        'max_v2', type=float, 
        help='Maximal p-value for correlation (e.g., 0.05)')
    parser.add_argument(
        'mod_type', type=str,
        help='Type of language model used for prediction (e.g., "roberta")')
    parser.add_argument(
        'mod_name', type=str,
        help='Language model used for prediction (e.g., "robert-base")')
    parser.add_argument(
        'scenario', type=str,
        help='Default separation ("defsep") or by data set ("datasep")')
    parser.add_argument(
        'test_ratio', type=float,
        help='Ratio of samples used for testing (e.g., 0.2), not training')
    parser.add_argument(
        'use_types', type=int, help='Use types for prediction (1) or not (0)')
    parser.add_argument(
        'out_path', type=str, help='Path to output file containing results')
    args = parser.parse_args()

    # get command line parameters
    coeff = args.coeff
    min_v1 = args.min_v1
    max_v2 = args.max_v2
    mod_type = args.mod_type
    mod_name = args.mod_name
    scenario = args.scenario
    test_ratio = args.test_ratio
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
    data = pd.read_csv(args.src_path, sep = ',')
    data = data.sample(frac=1, random_state=seed)
    data.columns = [
        'dataid', 'datapath', 'nrrows', 'nrvals1', 'nrvals2', 
        'type1', 'type2', 'column1', 'column2', 'method', 
        'coefficient', 'pvalue', 'time']

    # enrich column names if activated
    if args.use_types:    
        data = data.apply(add_type, axis=1)
    
    # filter data
    data = data[data['method']==coeff]
    nr_total = len(data.index)
    print(f'Nr. samples: {nr_total}')
    print('Sample from filtered data:')
    print(data.head())
    
    # label data
    def coefficient_label(row):
        """ Label column pair as correlated or uncorrelated. 
        
        Args:
            row: describes correlation between column pair.
        
        Returns:
            1 if correlated, 0 if not correlated. 
        """
        if abs(row['coefficient']) >= min_v1 and abs(row['pvalue']) <= max_v2:
            return 1
        else:
            return 0
    data['label'] = data.apply(coefficient_label, axis=1)
    
    # split into test and training
    if scenario == 'defsep':
        train, test = def_split(data, test_ratio, seed)
    elif scenario == 'datasep':
        train, test = ds_split(data, test_ratio)
    else:
        raise ValueError(f'Undefined scenario: {scenario}')
    
    train.columns = ['text_a', 'text_b', 'type1', 'type2', 'labels']
    test.columns = ['text_a', 'text_b', 'type1', 'type2', 'labels']
    print(train.head())
    print(test.head())
    
    # prepare loss scaling
    lab_counts = train['labels'].value_counts()
    nr_zeros = lab_counts.loc[0]
    nr_ones = lab_counts.loc[1]
    nr_all = float(len(train.index))
    weights = [nr_all/nr_zeros, nr_all/nr_ones]
    
    # train classification model
    s_time = time.time()
    model_args = ClassificationArgs(
        num_train_epochs=10, train_batch_size=100, eval_batch_size=100,
        overwrite_output_dir=True, manual_seed=seed, 
        evaluate_during_training=True, no_save=True)
    model = ClassificationModel(
        mod_type, mod_name, weight=weights,
        use_cuda = True, args=model_args)
    model.train_model(
        train_df=train, eval_df=test, acc=metrics.accuracy_score, 
        rec=metrics.recall_score, pre=metrics.precision_score, 
        f1=metrics.f1_score)
    training_time = time.time() - s_time
    
    test['length'] = test.apply(names_length, axis=1)
    test['nrtokens'] = test.apply(names_tokens, axis=1)
    
    # Initialize result file
    with open(args.out_path, 'w') as file:
        file.write(
            'coefficient,min_v1,max_v2,mod_type,mod_name,scenario,test_ratio,'
            'test_name,pred_method,lb,ub,f1,precision,recall,accuracy,mcc,'
            'prediction_time,training_time')
        
    # use simple baseline and model for prediction
    for m in [0, 1]:
        # use entire test set (redundant - for verification)
        test_name = f'{m}-final'
        log_metrics(
            coeff, min_v1, max_v2, mod_type, mod_name, scenario, 
            test_ratio, test, test_name, 0, 'inf', m, args.out_path)
    
        # test for data types
        for type1 in ['object', 'float64', 'int64', 'bool']:
            for type2 in ['object', 'float64', 'int64', 'bool']:
                sub_test = test.query(f'type1=="{type1}" and type2=="{type2}"')
                if sub_test.shape[0]:
                    test_name = f'Types{type1}-{type2}'
                    log_metrics(
                        coeff, min_v1, max_v2, mod_type, mod_name, scenario, 
                        test_ratio, sub_test, test_name, -1, -1, m, 
                        args.out_path)
    
        # test for different subsets
        for q in [(0, 0.25), (0.25, 0.5), (0.5, 1)]:
            qlb = q[0]
            qub = q[1]
            # column name length
            lb = test['length'].quantile(qlb)
            ub = test['length'].quantile(qub)
            sub_test = test[(test['length'] >= lb) & (test['length'] <= ub)]
            test_name = f'L{m}-{qlb}-{qub}'
            log_metrics(
                coeff, min_v1, max_v2, mod_type, mod_name, scenario, 
                test_ratio, sub_test, test_name, lb, ub, m, args.out_path)
            # number of tokens in column names
            lb = test['nrtokens'].quantile(qlb)
            ub = test['nrtokens'].quantile(qub)
            sub_test = test[(test['nrtokens'] >= lb) & (test['nrtokens'] <= ub)]
            test_name = f'N{m}-{qlb}-{qub}'
            log_metrics(
                coeff, min_v1, max_v2, mod_type, mod_name, scenario, 
                test_ratio, sub_test, test_name, lb, ub, m, args.out_path)