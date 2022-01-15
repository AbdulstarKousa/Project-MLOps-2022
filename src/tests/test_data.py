import itertools
import os
import sys

import pandas as pd
import pytest

from src import _PATH_DATA
from src.data.make_dataset import load_data, process_data, raw_data


def get_folder_structure(path):
  '''Returns the folder structure from the `path` variable.
  The returned object is an iterable. To be compared to
  the folder structure post-processing.
  '''
  return os.walk(path)

def iter_equal(items1, items2):
  '''`True` if iterators `items1` and `items2` contain equal items.
  Source: https://stackoverflow.com/questions/25216504/how-to-detect-that-two-python-iterators-yield-the-same-items'''
  return (items1 is items2) or \
          all(a == b for a, b in itertools.zip_longest(items1, items2, fillvalue=object()))

# raw data must remain the same before and after data processing
# i.e.: 
# - Must have the same folder and file structure
# - Must have the same size in megabytes

def prepare_data():
    '''Prepare data, output to folders data/raw/ and data/processed/'''
    raw_datasets = raw_data()
    process_data(raw_datasets)
    return raw_datasets

def train_test():
    prepare_data()
    train, test = load_data()
    train, test = pd.DataFrame(train), pd.DataFrame(test)

def test_same_struct():
    '''Tests whether the raw data before and after processing
    has the same folder and file structure.'''

    prepare_data()

    raw_path = _PATH_DATA + '/raw/'
    # Obtain file structure before processing
    pre_struct = get_folder_structure(raw_path)

    # Obtain file structure after processing
    post_struct = get_folder_structure(raw_path)

    # Compare the folder structures of pre- and post-processing
    # Structures must be the same
    is_equal = iter_equal(pre_struct, post_struct)

    assert is_equal is True

def test_same_size():
    # `raw_datasets` contains raw data before processing
    raw_datasets = prepare_data()
    raw_datasets_post = raw_data()
    size_before = sys.getsizeof(raw_datasets)
    size_after = sys.getsizeof(raw_datasets_post)

    assert size_before == size_after

def test_shape_eq():
    '''Tests whether the shape of the train and test datasets
    is as expected.'''
    train_shape = (25000,5)
    test_shape = (25000,5)

    prepare_data()
    train, test = load_data()

    assert train.shape==train_shape and test.shape==test_shape

def test_point_eq():
    '''Tests whether each datapoint has two fields: 
    an integer field 'label'
    and string field 'text'
    '''
    
    prepare_data()
    train, test = load_data()

    # Get data types of columns of the train and test datasets
    features_train, features_test = train.features, test.features

    # The data types of the train and test datasets
    dtypes_train = [feature.dtype for feature in features_train.values()]
    dtypes_test = [feature.dtype for feature in features_test.values()]

    # The data types which are expected
    types = ['list', 'list', 'int64', 'string', 'list']

    assert train.num_columns == test.num_columns == 5 and dtypes_train == dtypes_test == types

def rep(data):
    '''Checks if all labels are represented
    in the dataset `data`.'''
    labels = [0,1]
    data_rep = False
    # Iteratively check if all labels are represented
    for i in range(len(data)-1):
        row = data[i]
        label = row['label']
        if label in labels:
            try:
                # If label found, remove from list
                labels.pop(label)
            except IndexError:
                # List is empty, so all labels 
                # are represented
                data_rep = True
                break

    return data_rep

def test_label_rep():
    '''Checks if all labels are represented
    in the `train` and `test` datasets.'''
    prepare_data()
    train, test = load_data()

    train_rep, test_rep = rep(train), rep(test)
    assert train_rep == True, test_rep == True