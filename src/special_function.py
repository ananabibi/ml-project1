"""Decoding functions for the columns flagged Needs_special_functions = True
in data_columns.csv.

These BRFSS columns cannot be cleaned with a simple ValueMapping because the
value packs a unit (or a format) inside the number itself, e.g. 205 means
"5 times per week" and 130 means "1 hour 30 minutes". Every decoding function
takes one raw column (1D np.array) and returns a float column expressed in a
single unit, with NaN for "don't know", "refused" or invalid answers.

How the file is organised:
    1. Generic building blocks (_decode_prefixed, _decode_hours_minutes)
    2. One decoding function per *type* of encoding
    3. SPECIAL_FUNCTIONS: each function -> the list of columns it decodes
    4. apply_special_functions: applies everything to a data matrix

IMPORTANT: these functions expect the RAW codes. Run apply_special_functions
before the generic ValueMapping step (e.g. "888=0"), and skip these columns
in that step, otherwise codes like 888 are already replaced and the decoding
is wrong.

How to use it (from the project root, e.g. in projet.ipynb):

    from src.helpers import load_csv_data
    from src.special_function import (
        read_column_names, read_flagged_columns, special_columns,
        apply_special_functions,
    )

    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data")
    names = read_column_names("data/x_train.csv")

    # Optional sanity check: every flagged column should have a decoder
    missing = set(read_flagged_columns("data_columns.csv")) - set(special_columns())
    print("Flagged columns without a decoder:", missing)

    # Decode the special columns (train and test with the same function)
    x_train, train_names = apply_special_functions(x_train, names)
    x_test, test_names = apply_special_functions(x_test, names)

    # Then run the generic ValueMapping cleaning, skipping special_columns()

A single column can also be decoded on its own, e.g.
    weight_kg = weight_to_kg(x_train[:, names.index("WEIGHT2")])

To support a new column: add its name to the right list in SPECIAL_FUNCTIONS,
or write a new decoding function (raw 1D column -> decoded column) and add it
there as a new key.
"""

import csv
import numpy as np


# ---------------------------------------------------------------------------
# 1. Generic building blocks
# ---------------------------------------------------------------------------


def _decode_prefixed(col, unit_factors, zero_codes=(), nan_codes=(777, 999)):
    """
    Decodes a column where the first digit gives the unit and the last two
    digits give the count (e.g. 1xx = per day, 2xx = per week, ...).

    Example with unit_factors = {1: 1.0, 2: 1 / 7}:
        105 -> 5 * 1.0 = 5      (5 per day)
        214 -> 14 * 1/7 = 2     (14 per week = 2 per day)

    Args:
        col (np.array): raw column, shape (N,)
        unit_factors (dict): {first digit: factor} converting a count expressed
            in that unit into the target unit
        zero_codes (tuple): codes meaning "never" / "none", mapped to 0
        nan_codes (tuple): codes meaning "don't know" / "refused", mapped to NaN

    Returns:
        out (np.array): decoded column (float), shape (N,). Every code that is
            not described by the arguments (e.g. 555 when it is not a zero
            code) ends up as NaN.
    """
    # Work in float so NaN can be stored; start with everything unknown (NaN)
    # and only fill the entries we know how to decode
    col = np.asarray(col, dtype=float)
    out = np.full(col.shape, np.nan)

    # Split the code: hundreds digit = unit, last two digits = count
    # (e.g. 214 -> prefix 2, count 14). NaN inputs stay NaN.
    prefix = col // 100
    count = col % 100
    for digit, factor in unit_factors.items():
        # A count of 0 is not a valid answer inside a unit block
        # (special codes like 300 are handled through zero_codes)
        mask = (prefix == digit) & (count > 0)
        out[mask] = count[mask] * factor

    # Special codes are applied last so they override the unit decoding above
    # (e.g. 888 would otherwise be read as prefix 8, count 88)
    out[np.isin(col, zero_codes)] = 0.0
    out[np.isin(col, nan_codes)] = np.nan
    return out


def _decode_hours_minutes(col, nan_codes=(777, 999)):
    """
    Converts an H:MM coded column into minutes (130 = 1h30 = 90 min,
    45 = 45 min, 600 = 6h = 360 min).

    Args:
        col (np.array): raw column, shape (N,)
        nan_codes (tuple): codes mapped to NaN

    Returns:
        out (np.array): duration in minutes, shape (N,)
    """
    # H:MM is stored as a plain number: hundreds = hours, last two digits =
    # minutes (130 -> 1 h and 30 min). NaN inputs stay NaN.
    col = np.asarray(col, dtype=float)
    hours = col // 100
    minutes = col % 100
    out = hours * 60 + minutes

    # 777 / 999 are answer codes, and a "minutes" part >= 60 cannot be a time
    out[np.isin(col, nan_codes) | (minutes >= 60)] = np.nan
    return out


# ---------------------------------------------------------------------------
# 2. Decoding functions (one per type of encoding)
# ---------------------------------------------------------------------------


def weight_to_kg(col):
    """
    Codes: 50-999 = weight in pounds, 9000-9998 = 9000 + weight in kg,
    7777 = don't know, 9999 = refused.

    Args:
        col (np.array): raw column, shape (N,)

    Returns:
        out (np.array): weight in kg, shape (N,)
    """
    # Everything not matched below (7777, 9999, NaN, out of range) stays NaN
    col = np.asarray(col, dtype=float)
    out = np.full(col.shape, np.nan)

    # Answers given in pounds: convert with 1 lb = 0.45359237 kg
    pounds = (col >= 50) & (col < 1000)
    out[pounds] = col[pounds] * 0.45359237

    # Answers given in kg: remove the 9000 offset (9070 -> 70 kg).
    # 9999 (refused) is excluded by the strict upper bound.
    kilos = (col > 9000) & (col < 9999)
    out[kilos] = col[kilos] - 9000
    return out


def height_to_cm(col):
    """
    Codes: 200-711 = feet and inches written F'II (507 = 5 ft 7 in),
    9000-9998 = 9000 + height in cm, 7777 = don't know, 9999 = refused.

    Args:
        col (np.array): raw column, shape (N,)

    Returns:
        out (np.array): height in cm, shape (N,)
    """
    # Everything not matched below (7777, 9999, NaN, invalid codes) stays NaN
    col = np.asarray(col, dtype=float)
    out = np.full(col.shape, np.nan)

    # Answers given in feet/inches: hundreds digit = feet, last two = inches.
    # The inches part must be < 12, otherwise the code is invalid
    imperial = (col >= 200) & (col < 1000) & (col % 100 < 12)
    feet = col[imperial] // 100
    inches = col[imperial] % 100
    # Total inches, then 1 in = 2.54 cm (507 -> 67 in -> 170.18 cm)
    out[imperial] = (feet * 12 + inches) * 2.54

    # Answers given in cm: remove the 9000 offset (9175 -> 175 cm)
    metric = (col > 9000) & (col < 9999)
    out[metric] = col[metric] - 9000
    return out


def days_per_month(col):
    """
    Codes: 1xx = days per week, 2xx = days in the past 30 days,
    888 = none (mapped to 0), 777 / 999 = don't know / refused.

    Args:
        col (np.array): raw column, shape (N,)

    Returns:
        out (np.array): number of days over 30 days, between 0 and 30
    """
    # Days per week are scaled by 30/7 to be comparable with days per 30 days
    return _decode_prefixed(col, {1: 30 / 7, 2: 1.0}, zero_codes=(888,))


def times_per_day(col):
    """
    Codes: 1xx = per day, 2xx = per week, 3xx = per month,
    300 = less than once a month and 555 = never (both mapped to 0),
    777 / 999 = don't know / refused.

    Args:
        col (np.array): raw column, shape (N,)

    Returns:
        out (np.array): frequency in times per day, shape (N,)
    """
    # Per week -> divide by 7, per month -> divide by 30 (approximation)
    return _decode_prefixed(
        col, {1: 1.0, 2: 1 / 7, 3: 1 / 30}, zero_codes=(300, 555)
    )


def times_per_week(col):
    """
    Codes: 1xx = times per week, 2xx = times per month,
    888 = never (mapped to 0), 777 / 999 = don't know / refused.

    Args:
        col (np.array): raw column, shape (N,)

    Returns:
        out (np.array): frequency in times per week, shape (N,)
    """
    # Per month -> multiply by 7/30 to get times per week
    return _decode_prefixed(col, {1: 1.0, 2: 7 / 30}, zero_codes=(888,))


def minutes(col):
    """
    Codes: 1-759 and 800-959 = hours and minutes written H:MM,
    777 / 999 = don't know / refused.

    Args:
        col (np.array): raw column, shape (N,)

    Returns:
        out (np.array): duration in minutes, shape (N,)
    """
    # Thin wrapper so the column -> function table in SPECIAL_FUNCTIONS reads
    # by meaning ("minutes") rather than by implementation
    return _decode_hours_minutes(col)


def times_per_year(col):
    """
    Codes: 1xx = per day, 2xx = per week, 3xx = per month, 4xx = per year,
    888 = never (mapped to 0), 777 / 999 = don't know / refused.
    555 ("no feet" for FEETCHK2) means the question does not apply, so it is
    left as NaN rather than 0.

    Args:
        col (np.array): raw column, shape (N,)

    Returns:
        out (np.array): frequency in times per year, shape (N,)
    """
    # 555 is not listed in zero_codes, so _decode_prefixed leaves it as NaN
    return _decode_prefixed(
        col, {1: 365.0, 2: 52.0, 3: 12.0, 4: 1.0}, zero_codes=(888,)
    )


def duration_in_days(col):
    """
    Codes: 1xx = days, 2xx = weeks, 3xx = months, 4xx = years,
    777 / 999 = don't know / refused.
    555 is not a duration (most likely "never"); it is left as NaN here.
    Last year's team mapped it to the maximum value instead: to discuss.

    Args:
        col (np.array): raw column, shape (N,)

    Returns:
        out (np.array): duration in days, shape (N,)
    """
    # Weeks, months and years converted with 7, 30 and 365 days.
    # To map 555 to a value later, pass it through zero_codes or set it after.
    return _decode_prefixed(col, {1: 1.0, 2: 7.0, 3: 30.0, 4: 365.0})


def multiple_choice_flags(col, n_choices=4):
    """
    Multiple choice answer stored as the concatenation of the chosen options
    (12 = options 1 and 2, 21 = the same options in another order),
    7 / 9 = don't know / refused. A number like 12 has no meaningful order
    compared to 2, so the column is split into one binary column per option
    (one-hot style).

    Args:
        col (np.array): raw column, shape (N,)
        n_choices (int): number of possible options

    Returns:
        out (np.array): shape (N, n_choices), out[i, k] = 1 if option k+1 was
            chosen, 0 if not, NaN if the question was not answered
    """
    col = np.asarray(col, dtype=float)
    # One output column per option, NaN by default (= not answered)
    out = np.full((col.shape[0], n_choices), np.nan)
    # Rows with a real answer start at 0 ("option not chosen") ...
    answered = ~np.isnan(col) & ~np.isin(col, (7, 9))
    out[answered] = 0.0

    # ... then each digit of the code switches its option on (12 -> options
    # 1 and 2 -> out[i, 0] = out[i, 1] = 1). Digits outside 1..n_choices
    # are ignored.
    for i in np.where(answered)[0]:
        for digit in str(int(col[i])):
            k = int(digit) - 1
            if 0 <= k < n_choices:
                out[i, k] = 1.0
    return out


def met_value(col, max_met=12.8):
    """
    MET value of the activity (energy cost, 3.5 = moderate walking).
    The BRFSS file stores it with one implied decimal (35 = 3.5); our csv is
    already divided by 10, so we only rescale if values above the documented
    maximum appear, and we set anything out of [0, max_met] to NaN.

    Args:
        col (np.array): raw column, shape (N,)
        max_met (float): largest valid MET value

    Returns:
        out (np.array): MET value, shape (N,)
    """
    # Copy so the original data matrix is never modified in place
    col = np.asarray(col, dtype=float).copy()
    # If the column still has the implied decimal (max above 12.8), divide by 10
    if np.nanmax(col, initial=0) > max_met:
        col = col / 10
    # Values outside the valid MET range are treated as missing
    col[(col < 0) | (col > max_met)] = np.nan
    return col


# ---------------------------------------------------------------------------
# 3. Decoding function -> columns it applies to
# ---------------------------------------------------------------------------

# Table used by apply_special_functions: key = decoding function, value = names
# of the columns (as in the csv header) decoded by that function.
# Columns missing from the data are simply skipped.
SPECIAL_FUNCTIONS = {
    weight_to_kg: ["WEIGHT2"],
    height_to_cm: ["HEIGHT3"],
    days_per_month: ["ALCDAY5"],
    times_per_day: ["FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1"],
    times_per_week: ["EXEROFT1", "EXEROFT2", "STRENGTH"],
    minutes: ["EXERHMM1", "EXERHMM2"],
    times_per_year: ["BLDSUGAR", "FEETCHK2"],
    duration_in_days: ["LONGWTCH"],
    met_value: ["METVL11_", "METVL21_"],
}


def special_columns():
    """
    Lists the columns decoded in this file. Use it to skip them in the generic
    ValueMapping cleaning step.

    Returns:
        names (list): every column handled by SPECIAL_FUNCTIONS
    """
    return [name for cols in SPECIAL_FUNCTIONS.values() for name in cols]


# ---------------------------------------------------------------------------
# 4. Applying everything to the dataset
# ---------------------------------------------------------------------------


def read_column_names(csv_path):
    """
    Reads the header of x_train.csv / x_test.csv, without the Id column, so
    that names line up with the columns returned by load_csv_data.

    Args:
        csv_path (str): path to the csv file

    Returns:
        names (list): column names
    """
    # Only the first line is read, so this is fast even on the full dataset
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    # header[0] is "Id", which load_csv_data removes from x
    return [name.strip() for name in header[1:]]


def read_flagged_columns(columns_csv_path):
    """
    Reads data_columns.csv and returns the columns whose last field
    (Needs_special_functions) is True. Useful to check that
    SPECIAL_FUNCTIONS covers all of them.

    Args:
        columns_csv_path (str): path to data_columns.csv

    Returns:
        names (list): names of the flagged columns
    """
    names = []
    with open(columns_csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row and row[-1].strip() == "True":
                names.append(row[0].strip())
    return names


def apply_special_functions(x, column_names):
    """
    Replaces every special column of x by its decoded version. Columns that
    decode into several features (PCDMDECN) are expanded in place, so the
    other columns keep their order.

    Args:
        x (np.array): raw data, shape (N, D)
        column_names (list): names of the D columns of x

    Returns:
        x_new (np.array): decoded data, shape (N, D')
        new_names (list): names of the D' columns of x_new
    """
    # Decode every special column present in x
    # (index: column name -> position in x; decoded: column name -> result)
    index = {name: j for j, name in enumerate(column_names)}
    decoded = {}
    for func, cols in SPECIAL_FUNCTIONS.items():
        for name in cols:
            if name in index:
                decoded[name] = func(x[:, index[name]])

    # Rebuild the matrix in the original column order
    new_cols = []
    new_names = []
    for j, name in enumerate(column_names):
        if name not in decoded:
            # Normal column: copied unchanged (j : j + 1 keeps it 2D, (N, 1))
            new_cols.append(x[:, j : j + 1])
            new_names.append(name)
        elif decoded[name].ndim == 1:
            # Decoded into one column: replaces the raw one, same name
            new_cols.append(decoded[name][:, None])
            new_names.append(name)
        else:
            # Decoded into several columns (multiple_choice_flags): inserted
            # at the same place, named NAME_1, NAME_2, ...
            new_cols.append(decoded[name])
            new_names += [f"{name}_{k + 1}" for k in range(decoded[name].shape[1])]

    # Glue all the (N, k) blocks side by side into the final matrix
    return np.hstack(new_cols), new_names
