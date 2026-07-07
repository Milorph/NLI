from typing import Literal, Tuple
from datasets import load_dataset, Dataset
from common import resource_manager as rm


def evaluate_checkworthy() -> Dataset:
    """
    Dataset for Part (1) - evaluate check-worthy

    Label
    - 0: Non-Factual
    - 1: Unimportant Factual
    - 2: Check-worthy Factual

    Currently using ClaimBuster

    Returns:
        Dataset: 
    """

    dt = load_dataset('Nithiwat/claimbuster')
    
    return dt['train']


def find_logical_relationship() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Dataset for Part (3) - find logical relationship between two texts

    Label
    - 0: Entailment
    - 1: Neutral
    - 2: Contradiction

    Currently using MNLI

    Returns:
        Tuple: (Dataset, Dataset, Dataset)
            - Dataset: Train set
            - Dataset: Validation_matched set
            - Dataset: Validation_mismatched set
    """
    
    dt = load_dataset('nyu-mll/multi_nli') 

    return dt['train'], dt['validation_matched'], dt['validation_mismatched']

def find_logical_relationship_improve(num: Literal[1, 2, 3] = 1) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Dataset for Part (3) - find logical relationship between two texts (tricky)
    
    Label
    - 0: Entailment
    - 1: Neutral
    - 2: Contradiction

    Currently using ANLI

    Args:
        num (Literal[1, 2, 3]): select the subset

    Returns:
        Tuple: (Dataset, Dataset, Dataset)
            - Dataset: Train set
            - Dataset: Validation_matched set
            - Dataset: Validation_mismatched set
    """

    dt = load_dataset('facebook/anli')

    return dt[f'train_r{num}'], dt[f'dev_r{num}'], dt[f'test_r{num}']

_classify_evidence_filename = 'classify_evidence_dataset.p'

def classify_evidence() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Dataset for Part (2), (3), and (4) - classify if the evidence supports the claim text

    Label
    - SUPPORTS
    - NOT ENOUGH INFO
    - REFUTES

    Currently using FEVER

    Returns:
        Tuple: (Dataset, Dataset, Dataset)
            - Dataset: Train set
            - Dataset: Dev set
            - Dataset: Test set
    """

    dt = rm.load_resource(f'{_classify_evidence_filename}')
    if dt is not None:

        print('load by pickle')
        return dt['train'], dt['dev'], dt['test']
    
    main_dt = load_dataset('mteb/fever')
    corpus_dt = load_dataset('mteb/fever', 'corpus')
    queries_dt = load_dataset('mteb/fever', 'queries')

    print('start making corpus_map')
    corpus_map = {item['_id']: {'title': item['title'], 'text': item['text']} for item in corpus_dt['corpus']}
    
    print('start making queries_map')
    queries_map = {item['_id']: item['text'] for item in queries_dt['queries']}

    print('start adding columns')
    main_dt = main_dt.map(
        lambda x: {
            **x, 
            'corpus_title': (corpus_map.get(x['corpus-id']) or {}).get('title'),
            'corpus_text': (corpus_map.get(x['corpus-id']) or {}).get('text'),
            'query_text': queries_map.get(x['query-id'], None)
        },
        num_proc=4
    )

    print('save as pickle')
    rm.save_resource(main_dt, f'{_classify_evidence_filename}')

    return main_dt['train'], main_dt['dev'], main_dt['test']