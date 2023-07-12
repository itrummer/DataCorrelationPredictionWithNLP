'''
Created on Jul 12, 2023

@author: immanueltrummer
'''
import argparse
import datasets


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Path to input data')
    parser.add_argument('out_path', type=str, help='Path to output data')
    args = parser.parse_args()
    
    data = datasets.load_from_disk(args.in_path)
    
    def add_type(row):
        """ Adds data types to row names.
        
        Args:
            row: represents column pair
        
        Returns:
            row with expanded column names
        """
        row['column1'] = row['column1'] + ' ' + row['type1']
        row['column2'] = row['column2'] + ' ' + row['type2']
        return row
    
    data = data.map(add_type)
    data.save_to_disk(args.out_path)