'''
Created on Aug 22, 2021

@author: immanueltrummer
'''
import argparse
import datasets
from dp.nlp.common import CorrelationDS
from sklearn.utils.class_weight import compute_class_weight
import time
import torch
import transformers

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
        class_weight='balanced', classes=[0, 1], y=labels)
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
    
    start_s = time.time()
    trainer.train()
    total_s = time.time() - start_s
    print(f'Training time: {total_s}')
    
    model.save_pretrained(args.out_path)