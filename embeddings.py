from sentence_transformers import SentenceTransformer
import json
import torch
import pandas as pd
import os

def create_ordered_dict(unordered_dict):
    temp_dict = {'link':[],'description':[],'patient_dialog':[],'doctor_dialog':[],}
    for k,v in convo_dict.items():
        temp_dict['link'].append(v['link'])  
        temp_dict['description'].append(v['description'])
        temp_dict['patient_dialog'].append(v['dialog']['patient'])   
        temp_dict['doctor_dialog'].append(v['dialog']['doctor'])
    return temp_dict

def create_embeddings(sentences,batch_size,progress=True,multi=False):
    print("Checking if MPS backend for Torch is available:",torch.backends.mps.is_available())
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1',device='mps')
    # model = SentenceTransformer('all-MiniLM-L6-v2',device = 'mps')
    embeddings = model.encode(sentences,show_progress_bar=progress,batch_size=batch_size)
    return embeddings

if __name__ == "__main__":
    # base_path = 'datasets/processed/MedDialog_English/'
    master_df = pd.DataFrame(columns=('link','description','patient_dialog','doctor_dialog','patient_embeddings','doctor_embeddings'))
    base_path = 'datasets/post_processed/MedDialog_English/'
    for file in os.listdir(base_path):
        with open(base_path+file) as f:
            convo_dict = json.load(f)
        df = pd.DataFrame.from_dict(create_ordered_dict(convo_dict))
        master_df = pd.concat([master_df,df]).reset_index(drop=True)
    print("All files combined!")
    print('Creating patient embeddings:')
    master_df['patient_embeddings'] = create_embeddings(sentences=master_df['patient_dialog'],batch_size=64).tolist()
    print('Creating doctor embeddings:')
    master_df['doctor_embeddings'] = create_embeddings(sentences=master_df['doctor_dialog'],batch_size=64).tolist()
    print("Saving embeddings as a checkpoint to .pkl!")
    pd.to_pickle(master_df,'embeddings_large.pkl')
    
    