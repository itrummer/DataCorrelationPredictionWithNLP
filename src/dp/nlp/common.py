'''
Created on Aug 23, 2021

@author: immanueltrummer
'''
import torch.utils.data
import transformers

class CorrelationDS(torch.utils.data.Dataset):
    """ Represents training data for correlation prediction. """
    
    tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
    
    def __init__(self, columns_1, columns_2, labels):
        """ Initializes prediction for labeled column pairs.
        
        Args:
            columns_1: names of first columns
            columns_2: names of second columns
            labels: label indicating column correlation
        """
        self.encodings = self.tokenizer(
            columns_1, columns_2, 
            truncation=True, padding=True)
        self.labels = labels
    
    def __getitem__(self, idx):
        """ Return item at specified index.
        
        Args:
            idx: index of item to retrieve
        
        Returns:
            item at specified index
        """
        item = {key: torch.tensor(val[idx]) 
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        """ Returns number of items. """
        return len(self.labels)