'''
Created on Aug 22, 2021

@author: immanueltrummer
'''
import argparse
import datasets
import dp.nlp.common
import os
import torch
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from transformers import Trainer

# def predict(tokenizer, model, row):
    # """ Predict correlation for current row.
    #
    # Args:
        # tokenizer: tokenizer for model
        # model: correlation prediction model
        # row: describes a column pair
        #
    # Returns:
        # probability of correlation between columns
    # """
    # column_1 = row['column1']
    # column_2 = row['column2']
    # encodings = tokenizer(
        # column_1, column_2, 
        # truncation=True, padding=True, 
        # return_tensors='pt')
    # out = model(**encodings)
    # return torch.softmax(out.logits[0], 0)[1]

if __name__ == '__main__':
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Path to input data')
    parser.add_argument('model_path', type=str, help='Path to model')
    parser.add_argument('out_path', type=str, help='Path to output file')
    args = parser.parse_args()
    
    data = datasets.load_from_disk(args.in_path)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained(args.model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_data = dp.nlp.common.CorrelationDS(
        data['column1'], data['column2'], data['labels'])
    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions = trainer.predict(test_data)
    predictions = torch.Tensor(predictions.predictions).to(device)
    predictions = torch.softmax(predictions, -1)
    predictions = predictions[:,1]
    
    df = data.to_pandas()
    df['predictions'] = predictions.to('cpu')
    df.to_csv(args.out_path)