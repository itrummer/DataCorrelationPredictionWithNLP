'''
Created on Jun 14, 2023

@author: immanueltrummer
'''
import argparse
import pandas as pd


def similarity(column1, column2):
    """ Calculates Jaccard similarity between column names. 
    
    Args:
        column1: name of first column.
        column2: name of second column.
    
    Returns:
        Jaccard similarity between column names.
    """
    column1 = str(column1)
    column2 = str(column2)
    col1_parts = set(column1.split())
    col2_parts = set(column2.split())
    intersection_size = len(col1_parts.intersection(col2_parts))
    union_size = len(col1_parts) + len(col2_parts)
    return intersection_size/float(union_size)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='Path to input/output file')
    args = parser.parse_args()
    
    df = pd.read_csv(args.file_path)
    if 'jaccard' in df.columns:
        raise Exception('Jaccard column already exists!')
    
    jaccard = df.apply(lambda r:similarity(r['column1'], r['column2']), axis=1)
    df['jaccard'] = jaccard
    df.to_csv(args.file_path)