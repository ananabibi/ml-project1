import numpy as np
from special_function import apply_special_functions
from helpers import load_csv_data
import csv

DATA_PATH = "data"

TRAIN_PATH = "data/x_train.csv"
threshold = 0.4
# dictionnary of the columns to remove


def extract_headers():
    # read all the headers of x_train (x_train and x_test have the same headers)
    headers = np.genfromtxt(TRAIN_PATH, delimiter=",", max_rows=1, dtype=str)

    return headers[1:].tolist()  # we don't want the Id


def get_selected_col():
    # get the name of the kept_columns in data_columns
    data_path = "data_columns.csv"
    kept_headers = np.genfromtxt(
        data_path, delimiter=",", skip_header=1, usecols=0, dtype=str
    )
    return kept_headers.tolist()


def headers_to_indice():
    # find the index of the kept columns in x_train
    all_headers = extract_headers()
    selected_headers = get_selected_col()
    indices = []
    for head in selected_headers:
        index = all_headers.index(head)
        indices.append(index)

    return indices


def value_mapping(column_full_data: np.array, mapping: list, min_val: float, max_val: float):
    output = column_full_data
    
    for rep in mapping.split(";"):
        if not rep:  # if rep is empty
            continue
        value, replacement = rep.split("=")

        if replacement.strip() == "NaN":
            replacement = np.nan
        else:
            replacement = int(replacement)

        min_val = float(min_val) if min_val else -np.inf
        max_val= float(max_val) if max_val else np.inf

        if replacement < min_val or replacement > max_val:
            print("replacement value is outside min/max range")

        output = np.where(column_full_data == int(value), replacement, output)
    
    return output


def one_hot_encoding(column_full_data: np.array, min_val: float, max_val: float):
    # one-hot encoding will be done
    # columns = [np.isnan(column_full_data).astype(int)] we remove a column to avoid multicolinearity
    columns = []
    for value in range(int(min_val), int(max_val) + 1):
        new_col = column_full_data == int(value)
        columns.append(new_col)
    col_tab = np.column_stack(columns)
    return col_tab


def normalize_and_add_nan(
    column_full_data: np.array,
    min_val: float,
    max_val: float,
    norm_mean: float,
    norm_std: float,
):
    column_size = column_full_data.shape[0]
    norma_data = (column_full_data - norm_mean) / norm_std
    col_tab = np.zeros((column_size, 2))  # 2 columns, 1 for NaN, 1 for other values
    col_tab[:, 0] = np.where(np.isnan(column_full_data), 1, 0)
    col_tab[:, 1] = np.where(np.isnan(column_full_data), 0.0, norma_data)
    return col_tab


def preprocess_column(
    column_full_data: np.array,
    categorical: bool,
    mapping: list,
    min_val: float,
    max_val: float,
    norm_mean: float,
    norm_std: float,
):
    # preprocess a column of the data set

    column_full_data = value_mapping(column_full_data, mapping, min_val, max_val)  # new

    if categorical:
        column_full_data = one_hot_encoding(column_full_data, min_val, max_val)
    else:
        column_full_data = normalize_and_add_nan(
            column_full_data, min_val, max_val, norm_mean, norm_std
        )

    return column_full_data

def delete_too_many_nan(data, threshold): 
    mask=np.mean(np.isnan(data), axis=1) < threshold #mask is true or flase for each line 
    return data[mask,:]


def preprocess(train_data: np.array, test_data: np.array, meta_data: np.array): 
    train_reduced_data, test_reduced_data = dict(), dict()
    original_headers = extract_headers()

    train_data, headers = apply_special_functions(train_data, original_headers)
    test_data, _ = apply_special_functions(test_data, original_headers)

    #train_data = delete_too_many_nan(train_data,threshold)
    #test_data = delete_too_many_nan(test_data, threshold)

    if train_data.shape[1] != test_data.shape[1]: 
        raise ValueError("Train data and Test data have not the same number of columns")

    selected_cols = get_selected_col()
    for header, i in zip(headers, range(train_data.shape[1]), strict=True): 
        if header in selected_cols: 
            train_col_data=train_data[:,i]
            test_col_data=test_data[:,i]

            _, categorical, min_val, max_val, mapping, _ = meta_data[header]
            categorical = True if categorical == "True" else False

            norm_mean = np.mean(train_col_data[~np.isnan(train_col_data)])
            norm_std = np.std(train_col_data[~np.isnan(train_col_data)])

            train_reduced_data[header] = preprocess_column(train_col_data, categorical, mapping, min_val, max_val, norm_mean, norm_std)
            test_reduced_data[header] = preprocess_column(test_col_data, categorical, mapping, min_val, max_val, norm_mean, norm_std)

    train_data = np.hstack([train_reduced_data[h] for h in selected_cols])
    test_data = np.hstack([test_reduced_data[h] for h in selected_cols])

    return train_data, test_data

def load_metadata():
    metadata = {} 
    with open("data_columns.csv", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row:
                metadata[row[0]] = row[1:]
    return metadata

if __name__ == "__main__":
    x_train, x_test, _, _, _ = load_csv_data(DATA_PATH)
    x_train_prep, x_test_prep =preprocess(x_train, x_test, load_metadata())
