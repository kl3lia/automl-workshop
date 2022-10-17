from azureml.core import Run
import argparse
import mlflow
from random import random

import pandas as pd 
import numpy as np 
import os


# define functions
def prepare_data(df):
    #any preparation code
    #any preparation code
    #any preparation code
    #any preparation code
    run.log('Size',df.shape[0]) # it shows up under Metrics tab (UI)
    mlflow.log_metric("yeliz mlflow", random())# it shows up under Metrics tab

    mlflow.log_param("yeliz test mlflow ", "world")# it shows up under Params
 
    os.system(f"echo 'hello yeliz' > hello.txt")
    mlflow.log_artifact("hello.txt") # It creates hello.txt file under "outputs+logs" tab
    return df



run = Run.get_context()

parser=argparse.ArgumentParser()
parser.add_argument('--output_path', dest='output_path',required=True)
args=parser.parse_args()

train_ds=run.input_datasets['inference_Classification_dataset']
df=train_ds.to_pandas_dataframe()

df=prepare_data(df)

df.to_csv(os.path.join(args.output_path,"prepped_inference_data_classification.csv"))

print(f"Wrote prepped data to {args.output_path}/prepped_inference_data.csv")












