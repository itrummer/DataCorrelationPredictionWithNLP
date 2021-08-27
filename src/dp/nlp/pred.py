'''
Created on Aug 22, 2021

@author: immanueltrummer
'''
import argparse
import datasets
import dp.nlp.common
import os
import time
import torch
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from transformers import Trainer

if __name__ == '__main__':
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Path to input data')
    parser.add_argument('model_path', type=str, help='Path to model')
    parser.add_argument('out_path', type=str, help='Path to output file')
    args = parser.parse_args()
    
    data = datasets.load_from_disk(args.in_path)
    df = data.to_pandas()
    nr_pairs = df.shape[0]
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    p_model = RobertaForSequenceClassification.from_pretrained(args.model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_data = dp.nlp.common.CorrelationDS(
        data['column1'], data['column2'], data['labels'])
    trainer = Trainer(model=p_model, tokenizer=tokenizer)
    
    start_s = time.time()
    predictions = trainer.predict(test_data)
    total_s = time.time() - start_s
    avg_s = total_s / nr_pairs
    df['ptime'] = avg_s
    print(f'Prediction took {total_s} seconds (avg: {avg_s}).')
    
    predictions = torch.Tensor(predictions.predictions).to(device)
    predictions = torch.softmax(predictions, -1)
    predictions = predictions[:,1]
    df['predictions'] = predictions.to('cpu')
    
    df.to_csv(args.out_path)