import pickle
from pathlib import Path


_base_path = Path(__file__).resolve().parent.parent
_resource_path = _base_path/'resource'

def load_resource(filename: str):

    try:

        with open(f'{_resource_path}/{filename}', 'rb') as f:

            return pickle.load(f)
        
    except FileNotFoundError:

        return None
    
def save_resource(data, filename: str):

    with open(f'{_resource_path}/{filename}', 'wb') as f:

        pickle.dump(data, f)