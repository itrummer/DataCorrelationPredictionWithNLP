'''
Created on Aug 22, 2021

@author: immanueltrummer
'''
import argparse
from datasets import Dataset, Features, Value, ClassLabel
import pandas as pd
import random

def load_data(in_path, cor_type):
    """ Loads data from .csv file.
    
    Args:
        in_path: path to input file
        cor_type: correlation type
    
    Args:
        a data frame
    """
    df = pd.read_csv(in_path, sep=',')
    df.columns = [
        'dataid', 'datapath', 'nrrows', 'nrvals1', 'nrvals2',
        'type1', 'type2', 'column1', 'column2', 'method',
        'coefficient', 'pvalue', 'time']
    return df.query(f'method=="{cor_type}"')

def split_data(df):
    """ Splits data set into training and test data.
    
    Args:
        df: data frame
    
    Returns:
        pair of training and test samples
    """
    data_ids = df['dataid'].unique()
    random.shuffle(data_ids)
    nr_data_sets = len(data_ids)
    nr_train_sets = round(nr_data_sets * 0.8)
    train_ds = list(data_ids[:nr_train_sets])
    test_ds = list(data_ids[nr_train_sets:])
    train_samples = df.query(f'dataid in {train_ds}')
    test_samples = df.query(f'dataid in {test_ds}')
    return train_samples, test_samples

def is_correlated(row):
    """ Check whether two columns are considered correlated.
    
    Args:
        row: a row describing relationship between two columns
    
    Returns:
        true iff the two columns are correlated
    """
    if row['coefficient'] >= 0.95 and row['pvalue'] <= 0.05:
        return 1
    else:
        return 0

def labeled_data(df):
    """ Create data set and label data.
    
    Args:
        df: data frame without labels
    
    Returns:
        labeled data set in HG format
    """
    labels = df.apply(is_correlated, axis='columns')
    print(df.info())
    features = Features({
        'dataid':Value('int64'),
        'nrrows':Value('int64'),
        'coefficient':Value('float64'),
        'pvalue':Value('float64'),
        'column1':Value('string'), 
        'column2':Value('string'), 
        'labels':ClassLabel(
            num_classes=2, 
            names=['Uncorrelated', 'Correlated'])})
    df = pd.DataFrame({
        'dataid':df['dataid'],
        'nrrows':df['nrrows'],
        'coefficient':df['coefficient'],
        'pvalue':df['pvalue'],
        'column1':df['column1'], 
        'column2':df['column2'], 
        'labels':labels})
    return Dataset.from_pandas(df, features)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Path to input file')
    args = parser.parse_args()
    df = load_data(args.in_path, 'pearson')
    train_samples, test_samples = split_data(df)
    
    train_ds = labeled_data(train_samples)
    train_ds = train_ds.shuffle()
    train_ds.save_to_disk('data/train.ds')
    test_ds = labeled_data(test_samples)
    test_ds.save_to_disk('data/test.ds')