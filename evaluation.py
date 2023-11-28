import pickle
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import BERTScorer
import time
import evaluate

def tokenize(text):
    return text.split()

def bleu_score(test_df, evaluation_type):

    if evaluation_type == 'context':
        reference = test_df['context']
    elif evaluation_type == 'doctor_dialog':
        reference = test_df['doctor_dialog']

    response_tokens = [tokenize(r) for r in test_df['response']]
    reference_tokens = [tokenize(r) for r in reference]

    scores = []
    count = 0
    smoothing_function = SmoothingFunction().method1
    
    for i in range(len(response_tokens)):
        try:
            # if test_df['context'] is not empty calculate bleu score else skip the sample
            if test_df['context'][i] != '[]':
                # Calculate BLEU score
                score = corpus_bleu([reference_tokens[i]], [response_tokens[i]], smoothing_function=smoothing_function)
                scores.append(score)
                count += 1
            else:
                continue
        except KeyError as e:
            continue
            
    print(f"BLEU score calculated for {count} out of {len(response_tokens)} samples")

    return scores

def bert_score(test_df, evaluation_type):

    if evaluation_type == 'context':
        reference = test_df['context']
    elif evaluation_type == 'doctor_dialog':
        reference = test_df['doctor_dialog']

    scorer = BERTScorer(model_type='bert-base-uncased')
    reference_dialogs = []
    candidate_responses = []
    for i in range(len(test_df)):
        try:
            # if test_df['context'] is not empty add doctor dialog and response to list else skip the sample
            if test_df['context'][i] != '[]':
                reference_dialogs.append(reference[i])
                candidate_responses.append(test_df['response'][i])
            else:
                continue
        except KeyError as e:
            continue

    P, R, F1 = scorer.score(candidate_responses, reference_dialogs)
    print(f"BERT score calculated for {len(F1)} out of {len(test_df)} samples")
    return P, R, F1

def rouge_score(test_df, evaluation_type):
    if evaluation_type == 'context':
        reference = test_df['context']
    elif evaluation_type == 'doctor_dialog':
        reference = test_df['doctor_dialog']

    reference_dialogs = []
    candidate_responses = []
    for i in range(len(test_df)):
        try:
            # if test_df['context'] is not empty add doctor dialog and response to list else skip the sample
            if test_df['context'][i] != '[]':
                reference_dialogs.append(reference[i])
                candidate_responses.append(test_df['response'][i])
            else:
                continue
        except KeyError as e:
            continue
    
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=test_df['response'], references=reference)
    print(f"ROUGE score calculated for {len(results)} out of {len(test_df)} samples")

    return results

def run_evaluation(evaluation_type, df):
    print(f"Evaluation type: {evaluation_type}")
    # Calculate BLEU score for test set
    start_time = time.time()
    b_scores = bleu_score(df, evaluation_type=evaluation_type)         
    # Save to text file
    if evaluation_type == 'context':
        with open('datasets/bleu_scores_context.txt', 'w') as f:
            for item in b_scores:
                f.write("%s\n" % item)
    elif evaluation_type == 'doctor_dialog':
        with open('datasets/bleu_scores_ground_truth.txt', 'w') as f:
            for item in b_scores:
                f.write("%s\n" % item)

    avg_b_score = sum(b_scores)/len(b_scores)
    end_time = time.time()
    print(f"Time taken to calculate BLEU score: {end_time-start_time}")

    # Calculate BERT score
    start_time = time.time()
    P, R, F1 = bert_score(df, evaluation_type=evaluation_type)
    # Save F1 scores text file
    if evaluation_type == 'context':
        with open('datasets/bert_scores_context.txt', 'w') as f:
            for item in F1:
                f.write("%s\n" % item)
    elif evaluation_type == 'doctor_dialog':
        with open('datasets/bert_scores_ground_truth.txt', 'w') as f:
            for item in F1:
                f.write("%s\n" % item)
    print(f"Time taken to calculate BERT score: {end_time-start_time}")

    # Calculate ROUGE score
    start_time = time.time()
    r_score = rouge_score(df, evaluation_type=evaluation_type)
    # Save ROUGE scores text file
    if evaluation_type == 'context':
        with open('datasets/rouge_scores_context.txt', 'w') as f:
            for item in r_score:
                f.write("%s\n" % item)
    elif evaluation_type == 'doctor_dialog':
        with open('datasets/rouge_scores_ground_truth.txt', 'w') as f:
            for item in r_score:
                f.write("%s\n" % item)
    end_time = time.time()
    print(f"Time taken to calculate ROUGE score: {end_time-start_time}")

    print('\n')
    print('--'*20)
    print(f"Average BLEU Score: {avg_b_score:.4f}")
    print(f"BERT Score (Mean F1): {F1.mean():.4f}")
    print(f"ROUGE score: {r_score['rougeL']:0.4f}")

    print('Evaluation completed for: ', evaluation_type)
    print('--'*20)
    print('--'*20)
    print('\n')



if __name__ == "__main__":
    test_samples = pd.read_pickle("datasets/test_samples.pkl").reset_index(drop=True)
    generated_responses = pd.read_csv("datasets/responses_all.csv")

    doctor_dialogs = test_samples['doctor_dialog']
    responses = generated_responses['response']
    context = generated_responses['context']

    df = pd.DataFrame({'doctor_dialog': doctor_dialogs, 'response': responses, 'context': context})

    # run evaluation for ground truth and context
    run_evaluation(evaluation_type='doctor_dialog', df=df)
    run_evaluation(evaluation_type='context', df=df)