import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from azureml.core.model import Model
from azureml.core import Run#, _OfflineRun

model = None

def init():
    global model
    print("Loading model from model registery") # it will be seen underdriver logs
    run = Run.get_context()
    #ws = Workspace.from_config() if type(run) == _OfflineRun else run.experiment.workspace
    ws = Workspace.from_config() if not hasattr(run, 'experiment') else run.experiment.workspace

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model Name")
    args=parser.parse_args()

    path=Model.get_model_path(args.model_name, _workspace=ws)
    model_path = os.path.join(path, 'model.pkl')
    print(f"Model Path: {model_path}")
    model = joblib.load(model_path)


def run(file_list):

    print(f"Files to process: {file_list}")
    print("Generating predictions...")
    #scoring code
    #
    # model.forecast_quantiles()
    # model.predict()
    #
    #scoring code
    
    all_results = []
    return all_results
    
