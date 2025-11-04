import lmdb
import numpy as np
import os
import pickle
import pandas as pd
import argparse

class Chemformer:
    name: str = "Chemformer"
    description: str = 'Input the question, returns answers. Note: Please input the SMILES representation in the form of "SMILES"'
    def __init__(
        self,
        **tool_args
    ):  
        import json
        with open(f"./Result/Stacking/ReactionPrediction/Chemformer.json", 'r', encoding='utf-8') as f:
            data_test = json.load(f)
        with open(f"./Result/Stacking/ReactionPrediction/Chemformer_train.json", 'r', encoding='utf-8') as f:
            data_train = json.load(f)    
        data = data_test+data_train
        
        self.query_data = {i['target_smiles']+'>>___':i['answer'] for i in data}
        
    def _run(self, query: str,**tool_args) -> str:
        # Extract reaction equation from query
        if "the following Chemical reaction equation:" in query:
            query = query.split("the following Chemical reaction equation:")[-1].strip()
        else:
            # If the expected format is not found, use the query as is
            query = query.strip()
        
        if query in self.query_data:
            return self.query_data[query],0
        else:
            # Return a more helpful message instead of printing "error"
            return f"No matching reaction found for: {query}",0
        
    def __str__(self):
        return "Chemformer"

    def __repr__(self):
        return self.__str__()
    
    def wo_run(self,query):
        # Extract reaction equation from query
        if "the following Chemical reaction equation:" in query:
            query = query.split("the following Chemical reaction equation:")[-1].strip()
        else:
            # If the expected format is not found, use the query as is
            query = query.strip()
        
        if query in self.query_data:
            return self.query_data[query],0
        else:
            # Return a more helpful message instead of printing "error"
            return f"No matching reaction found for: {query}",0


