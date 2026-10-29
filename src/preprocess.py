"""Preprocessing pipeline: raw BRFSS survey codes -> clean numeric matrix tx.

The pipeline is split into two functions so that no information leaks from
validation / test data into training:

    stats = fit_preprocess(x_train, names)   # learns everything on train only
    tx_train = transform(x_train, stats)
    tx_test = transform(x_test, stats)

Steps (in this order):
    1. drop administrative / identifier columns and columns with no metadata
    2. decode the 20 coded columns with the special functions (raw codes!)
    3. apply ValueMapping + [Min, Max] of data_columns.csv to the other columns
    4. drop columns with too many NaN (ratio fitted on train)
    5. categorical columns  -> one-hot encoding (categories seen in train)
       numerical columns    -> optional log1p, median imputation, standardization
    6. optional "was missing" indicators for numerical columns
    7. drop constant columns, add the bias column
"""

import csv
import numpy as np

from src.special_function import apply_special_functions, special_columns

# Columns that describe the interview, not the person (ids, dates, phone).
ADMIN_COLUMNS = {
    "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE",
    "SEQNO", "_PSU", "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES",
    "CELLFON3", "LADULT", "CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2",
    "CCLGHOUS", "CSTATE", "LANDLINE", "QSTVER", "QSTLANG", "_STSTR",
    "_STRWT", "_RAWRAKE", "_WT2RAKE", "_LLCPWT", "_CLLCPWT", "_DUALUSE",
    "_DUALCOR",
}

DEFAULT_CONFIG = {
    "max_nan_ratio": 0.9,       # drop columns with more NaN than this (train)
    "max_categories": 15,       # one-hot only categorical columns up to this size
    "log1p": True,              # log1p on heavy-tailed non-negative columns
    "missing_indicators": True, # add a 0/1 "was missing" column per numeric column
}


def read_metadata(path):
    """
    Reads data_columns.csv.

    Args:
        path (str): path to data_columns.csv

    Returns:
        meta (dict): name -> {"categorical": bool, "min": float or None,
            "max": float or None, "mapping": {code: value}, "special": bool}
    """
    meta = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header (field names have surrounding spaces)
        for row in reader:
            if not row or not row[0].strip():
                continue
            row = [c.strip() for c in row] + [""] * 7
            name, _, is_cat, vmin, vmax, mapping, special = row[:7]
            codes = {}
            for item in mapping.split(";"):
                if "=" in item:
                    code, value = item.split("=")
                    codes[float(code)] = np.nan if value.strip() == "NaN" else float(value)
            meta[name] = {
                "categorical": is_cat == "True",
                "min": float(vmin) if vmin else None,
                "max": float(vmax) if vmax else None,
                "mapping": codes,
                "special": special == "True",
            }
    return meta


def _clean(x, names, meta):
    """
    Steps 1-3: drops admin / unknown columns, decodes special columns and
    applies the ValueMapping and [Min, Max] range. Stateless (no fitting).

    Args:
        x (np.array): raw data, shape (N, D)
        names (list): column names of x
        meta (dict): output of read_metadata

    Returns:
        x_clean (np.array): cleaned data, shape (N, D')
        clean_names (list): names of the D' columns
    """
    # 1. keep only columns described in data_columns.csv, minus admin ones
    keep = [j for j, n in enumerate(names) if n in meta and n not in ADMIN_COLUMNS]
    x = x[:, keep].astype(float)
    names = [names[j] for j in keep]

    # 2. special functions work on the raw codes
    x, names = apply_special_functions(x, names)
    special = set(special_columns())

    # 3. ValueMapping and valid range for every other column
    for j, n in enumerate(names):
        if n in special or n not in meta:
            continue
        col = x[:, j].copy()
        raw = x[:, j]
        for code, value in meta[n]["mapping"].items():
            col[raw == code] = value
        lo, hi = meta[n]["min"], meta[n]["max"]
        if lo is not None:
            col[col < lo] = np.nan
        if hi is not None:
            col[col > hi] = np.nan
        x[:, j] = col
    return x, names


def fit_preprocess(x_train, names, meta_path="data_columns.csv", config=None):
    """
    Learns every preprocessing statistic on the training data only.

    Args:
        x_train (np.array): raw training data, shape (N, D)
        names (list): column names of x_train
        meta_path (str): path to data_columns.csv
        config (dict): options, see DEFAULT_CONFIG

    Returns:
        stats (dict): everything transform() needs
    """
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    meta = read_metadata(meta_path)
    x, cnames = _clean(x_train, names, meta)

    # 4. drop columns with too many missing values
    nan_ratio = np.isnan(x).mean(axis=0)
    cols = [j for j in range(x.shape[1]) if nan_ratio[j] <= cfg["max_nan_ratio"]]

    cat_cols, num_cols = [], []
    for j in cols:
        n = cnames[j]
        is_cat = n in meta and meta[n]["categorical"]
        values = np.unique(x[~np.isnan(x[:, j]), j])
        if is_cat and 2 < len(values) <= cfg["max_categories"]:
            cat_cols.append((j, values))  # one-hot with train categories
        else:
            num_cols.append(j)            # binary categories stay numeric

    # 5. numeric statistics: log1p for heavy tails, median, mean, std
    xn = x[:, num_cols]
    log_mask = np.zeros(len(num_cols), dtype=bool)
    if cfg["log1p"]:
        mins = np.nanmin(xn, axis=0)
        p99 = np.nanpercentile(xn, 99, axis=0)
        med = np.nanmedian(xn, axis=0)
        # non-negative columns whose top 1% is far above the median
        log_mask = (mins >= 0) & (p99 > 10 * np.maximum(med, 1))
        xn = xn.copy()
        xn[:, log_mask] = np.log1p(xn[:, log_mask])
    medians = np.nanmedian(xn, axis=0)
    filled = np.where(np.isnan(xn), medians, xn)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std[std == 0] = 1.0

    stats = {
        "meta": meta, "names": list(names), "cfg": cfg, "clean_names": cnames,
        "num_cols": num_cols, "cat_cols": cat_cols, "log_mask": log_mask,
        "medians": medians, "mean": mean, "std": std,
        # 6. indicators only where something is actually missing in train
        "ind_cols": [k for k, j in enumerate(num_cols) if nan_ratio[j] > 0.01]
        if cfg["missing_indicators"] else [],
    }

    # 7. find constant output columns once, on the training data
    tx = _assemble(x, stats)
    stats["keep_out"] = np.concatenate([[True], tx[:, 1:].std(axis=0) > 0])
    return stats


def _assemble(x_clean, stats):
    """
    Builds the feature matrix from cleaned data (steps 5-6 + bias).

    Args:
        x_clean (np.array): output of _clean, shape (N, D')
        stats (dict): output of fit_preprocess

    Returns:
        tx (np.array): features with a leading bias column
    """
    xn = x_clean[:, stats["num_cols"]].copy()
    missing = np.isnan(xn)
    xn[:, stats["log_mask"]] = np.log1p(np.clip(xn[:, stats["log_mask"]], 0, None))
    xn = np.where(missing, stats["medians"], xn)
    xn = (xn - stats["mean"]) / stats["std"]

    blocks = [np.ones((x_clean.shape[0], 1)), xn]
    for j, values in stats["cat_cols"]:
        # one 0/1 column per category seen in train; NaN / unseen -> all zeros
        blocks.append((x_clean[:, j][:, None] == values[None, :]).astype(float))
    if stats["ind_cols"]:
        blocks.append(missing[:, stats["ind_cols"]].astype(float))
    return np.hstack(blocks)


def transform(x, stats):
    """
    Applies the fitted preprocessing to any data (train, validation or test).

    Args:
        x (np.array): raw data, shape (N, D), same columns as at fit time
        stats (dict): output of fit_preprocess

    Returns:
        tx (np.array): preprocessed features with bias column, shape (N, F)
    """
    x_clean, _ = _clean(x, stats["names"], stats["meta"])
    tx = _assemble(x_clean, stats)
    if "keep_out" in stats:
        tx = tx[:, stats["keep_out"]]
    return tx
