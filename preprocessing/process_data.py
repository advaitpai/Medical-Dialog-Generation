import pandas as pd
import spacy
import tqdm
import multiprocessing as mp
import json
import os
import argparse as ap

nlp = spacy.load('en_core_web_trf') 
hm_d1_dict = dict()
names_list = []
errors = []
def create_dict(path):
        with open(path,mode='r',encoding='utf-8') as f:
            hm_d1_str = [x.strip() for x in f.readlines() if x.strip() != '']
            # hm_d1_str = hm_d1_str[:20] # For testing purposes
            hmm_d1_dict = dict()
            flag = 0 #0 - idx, 1 - link, 2 - description, 3 - dialog, 4 - patient, 5 - doctor 
            for s in tqdm.tqdm(hm_d1_str):
                # Logics for flags
                if s.startswith("id="):
                    flag = 0
                    idx = s[3:]
                    hm_d1_dict[int(idx)] = dict()
                    hm_d1_dict[int(idx)]['link'] = ''
                    hm_d1_dict[int(idx)]['description'] = ''
                    hm_d1_dict[int(idx)]['dialog'] = dict()
                    hm_d1_dict[int(idx)]['dialog']['patient'] = []
                    hm_d1_dict[int(idx)]['dialog']['doctor'] = []
                    continue
                elif s.startswith("https"):
                    flag = 1
                    hm_d1_dict[int(idx)]['link'] = s 
                    continue  
                elif "Description" in s:
                    flag = 2
                    continue
                elif "Dialogue" in s:
                    flag = 3
                    continue
                elif "Patient" in s:
                    flag = 4
                    continue   
                elif "Doctor" in s:
                    flag = 5
                    continue
                # Logics for appends
                elif flag == 2:
                    hm_d1_dict[int(idx)]['description'] += s
                    continue
                elif flag == 4:
                    # Redact names
                    names = [x.text for x in nlp(s).ents if x.label_ == 'PERSON']
                    if len(names)>0:
                        for name in names:
                            s = s.replace(name,'[NAME]')
                        names_list.append(names)
                    hm_d1_dict[int(idx)]['dialog']['patient'].append(s)
                    continue
                elif flag == 5:
                    # Redact names
                    names = [x.text for x in nlp(s).ents if x.label_ == 'PERSON']
                    if len(names)>0:
                        for name in names:
                            s = s.replace(name,'[NAME]')
                        names_list.append(names)
                    hm_d1_dict[int(idx)]['dialog']['doctor'].append(s)
                    continue

if __name__ == '__main__':
    # Get file path from args
    parser = ap.ArgumentParser()
    parser.add_argument('-f','--file',type=str,help='Path to the file')
    args = parser.parse_args()
    path_in = args.file

    # Create dictionary
    create_dict(path_in)

    ## Write hm_d1_dict to a json fie
    path = 'datasets/processed/'+path_in.split('/')[-2]+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = path_in.split('/')[-1].split('.')[0]+'.json'
    with open(path+file_name,mode='w',encoding='utf-8') as f:
        json.dump(hm_d1_dict,f,indent=4)            

