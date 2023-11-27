import pickle
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import BERTScorer
import tqdm
import time

def tokenize(text):
    return text.split()

def bleu_score(test_df):

    # tokenize doctor dialogs and responses
    response_tokens = [tokenize(r) for r in test_df['response']]
    docdialogs_tokens = [tokenize(d) for d in test_df['doctor_dialog']]

    scores = []
    count = 0
    smoothing_function = SmoothingFunction().method1
    
    for i in range(len(response_tokens)):
        try:
            # if test_df['context'] is not empty calculate bleu score else skip the sample
            if test_df['context'][i] != '[]':
                # Calculate BLEU score
                score = corpus_bleu([docdialogs_tokens[i]], [response_tokens[i]], smoothing_function=smoothing_function)
                scores.append(score)
                count += 1
            else:
                continue
        except KeyError as e:
            continue
            
    print(f"BLEU score calculated for {count} out of {len(response_tokens)} samples")

    return scores

def bert_score(test_df):
    scorer = BERTScorer(model_type='bert-base-uncased')
    reference_dialogs = []
    candidate_responses = []
    for i in range(len(test_df)):
        try:
            # if test_df['context'] is not empty add doctor dialog and response to list else skip the sample
            if test_df['context'][i] != '[]':
                reference_dialogs.append(test_df['doctor_dialog'][i])
                candidate_responses.append(test_df['response'][i])
            else:
                continue
        except KeyError as e:
            continue

    P, R, F1 = scorer.score(candidate_responses, reference_dialogs)
    print(f"BERT score calculated for {len(F1)} out of {len(test_df)} samples")
    return P, R, F1

def perplexity_score(context, response):
    return


if __name__ == "__main__":

    test_samples = pd.read_pickle("datasets/test_samples.pkl").reset_index(drop=True)
    # extract 0 to100 and and 200 to 300 samples
    test_samples_1 = test_samples.iloc[0:100]
    test_samples_2 = test_samples.iloc[200:300]
    test_samples = pd.concat([test_samples_1, test_samples_2], ignore_index=True)
    doctor_dialogs = test_samples['doctor_dialog']
    generated_responses = pd.read_csv("datasets/responses_all.csv")
    responses = generated_responses['response']
    context = generated_responses['context']

    # create dataframe having columns doctor dialogs and responses and context
    df = pd.DataFrame({'doctor_dialog': doctor_dialogs, 'response': responses, 'context': context})

    print('length of actual doctor dialogs: ', len(doctor_dialogs))
    print('length of responses: ', len(responses))
    print('\n')


    # Calculate BLEU score for test set
    start_time = time.time()
    b_scores = bleu_score(df)         
    # save to text file
    with open('datasets/bleu_scores.txt', 'w') as f:
        for item in b_scores:
            f.write("%s\n" % item)

    avg_b_score = sum(b_scores)/len(b_scores)
    end_time = time.time()
    print(f"Time taken to calculate BLEU score: {end_time-start_time}")

    # Calculate BERT score
    start_time = time.time()
    P, R, F1 = bert_score(df)
    # save F1 scores text file
    with open('datasets/bert_scores.txt', 'w') as f:
        for item in F1:
            f.write("%s\n" % item)
    end_time = time.time()
    print(f"Time taken to calculate BERT score: {end_time-start_time}")
    
    print('\n')
    print('--'*20)
    print(f"Average BLEU Score: {avg_b_score}")
    print(f"BERT Score (Mean F1): {F1.mean():.4f}")


    # Calculate perplexity score

    # # Calculate ROUGE score for test set
    # r_score = rouge_score(context, responses)
    # print(f"ROUGE Score: {r_score}")