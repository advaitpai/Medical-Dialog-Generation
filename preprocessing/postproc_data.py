import pandas as pd
import tqdm
import multiprocessing as mp
import json
import os
import argparse as ap

hm_d1_dict = dict()
names_list = []
errors = []
def combine_convo(path):
    with open(path,mode='r') as f:
        hm_d1_dict = json.load(f)
        for k in tqdm.tqdm(hm_d1_dict.keys()):
            hm_d1_dict[k]["dialog"]["patient"] = " ".join(hm_d1_dict[k]["dialog"]["patient"])
            hm_d1_dict[k]["dialog"]["doctor"] = " ".join(hm_d1_dict[k]["dialog"]["doctor"])
    return hm_d1_dict

if __name__ == '__main__':
    # Get file path from args
    parser = ap.ArgumentParser()
    parser.add_argument('-f','--file',type=str,help='Path to the file')
    args = parser.parse_args()
    path_in = args.file

    # Create dictionary
    hm_d1_dict = combine_convo(path_in)

    ## Write hm_d1_dict to a json fie
    path = 'datasets/post_processed/'+path_in.split('/')[-2]+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = path_in.split('/')[-1].split('.')[0]+'.json'
    with open(path+file_name,mode='w',encoding='utf-8') as f:
        json.dump(hm_d1_dict,f,indent=4)            

