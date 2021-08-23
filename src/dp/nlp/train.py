'''
Created on Aug 22, 2021

@author: immanueltrummer
'''
import argparse
import datasets
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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Path to input file')
    parser.add_argument('out_path', type=str, help='Path to output model')
    args = parser.parse_args()

    training_args = transformers.TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        report_to=None
    )
    
    model = transformers.RobertaForSequenceClassification.from_pretrained(
        'roberta-base')
    
    train_data = datasets.load_from_disk(args.in_path)
    print(train_data['column1'])

    train_data = CorrelationDS(
        train_data['column1'], train_data['column2'], train_data['labels'])
    trainer = transformers.Trainer(
        model=model, args=training_args, 
        train_dataset=train_data)
    trainer.train()