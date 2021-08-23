'''
Created on Aug 22, 2021

@author: immanueltrummer
'''
import argparse
import datasets
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import torch
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


class CorrelationTrainer(transformers.Trainer):
    """ Customized trainer using class weights. """
    
    def __init__(self, class_weights, *args, **kwargs):
        """ Initializes custom trainer.
        
        Args:
            class_weights: assigns a weight to each class
        """
        super().__init__(*args, **kwargs)
        self.class_weights = torch.Tensor(class_weights).to('cuda')
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """ Compute loss using class weights.
        
        Args:
            model: compute loss for this model
            inputs: batch of training samples
            return_outputs: whether we return outputs as well
        
        Returns:
            outputs and loss or loss alone
        """
        labels = inputs.pop('labels')
        batch_size = labels.shape[0]
        labels = torch.zeros(batch_size, 2, device='cuda').scatter_(
            1, labels.unsqueeze(1), 1)
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        loss = loss_fct(logits.view(-1, 2), labels.float().view(-1, 2))
        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Path to input file')
    parser.add_argument('out_path', type=str, help='Path to output model')
    args = parser.parse_args()
        
    train_data = datasets.load_from_disk(args.in_path)
    labels = train_data['labels']
    class_weights = compute_class_weight(
        'balanced', [0, 1], labels)
    train_data = CorrelationDS(
        train_data['column1'], train_data['column2'],
        train_data['labels'])

    training_args = transformers.TrainingArguments(
        output_dir='./results', num_train_epochs=5,
        per_device_train_batch_size=32, per_device_eval_batch_size=32,
        save_strategy=transformers.trainer_utils.IntervalStrategy.NO,
        report_to=None
    )
    
    model = transformers.RobertaForSequenceClassification.from_pretrained(
        'roberta-base')
    
    trainer = CorrelationTrainer(
        model=model, args=training_args,
        train_dataset=train_data, 
        class_weights=class_weights)
    trainer.train()
    model.save_pretrained(args.out_path)