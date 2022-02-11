import pandas as pd


def drop_invalid_rows(df: pd.DataFrame, key_cols):
    """
    Remove all rows that have blanks in any of the provided key columns.

    Args:
        df: Input data frame
        key_cols: Either a single column name or a list of column names

    Returns:
        Data frame with rows containing blanks in key columns removed.

    """
    if type(key_cols) == str:
        return df[(~df[key_cols].isnull()) & (~df[key_cols].isna())]
    elif type(key_cols) == list:
        return df[(~df[key_cols].isnull().any(axis=1)) & (~df[key_cols].isna().any(axis=1))]
    else:
        raise RuntimeError(f"key_col needs to be of type str or list")


def drop_unnamed_cols(df: pd.DataFrame):
    cols = [col for col in df.columns
            if not pd.api.types.is_string_dtype(type(col))
            or not col.lower().startswith("unnamed")]
    return df[cols]


def trim_all_strings(df: pd.DataFrame):
    """
    Applying strip to a number creates a NaN.
    Applying strip to a NaN creates the string "nan"

    Therefore, convert the complete series to str and trim.
    Replace the "nan" with NaN, so that isna() and similar work as expected.

    Args:
        df:

    Returns:

    """
    df.columns = df.columns.str.strip()
    for col in df:
        if pd.api.types.is_string_dtype(df[col].dtype):
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": None})


def all_strings_to_lower(df: pd.DataFrame):
    df.columns = df.columns.str.lower()
    for col in df:
        if pd.api.types.is_string_dtype(df[col].dtype):
            df[col] = df[col].str.lower()


def load_data_and_clean(file, sheet_name, key_cols, skiprows=0):
    """
    Standard procedure to load a sheet from the Excel that includes basic data cleaning steps.

    Args:
        file:
        sheet_name:
        key_cols: single column header or list of column headers
        skiprows:

    Returns:
        Data frame

    """
    df = pd.read_excel(file, sheet_name=sheet_name, skiprows=skiprows)
    df.columns = df.columns.astype(str)  # Numeric headers are a problem in the used functions
    df = drop_unnamed_cols(df)
    df = drop_invalid_rows(df, key_cols)
    trim_all_strings(df)
    all_strings_to_lower(df)
    if type(key_cols) == str:
        df.set_index(key_cols.lower(), inplace=True)
    elif type(key_cols) == list:
        df.set_index([c.lower() for c in key_cols], inplace=True)
    else:
        raise RuntimeError("key_cols must be of type str of list")
    if len(df.index.unique()) < len(df.index):
        tmp = df.reset_index()[key_cols.lower()]
        if type(key_cols) == str:
            print(tmp.loc[tmp.duplicated()])
        elif type(key_cols) == list:
            print(tmp.loc[tmp.duplicated(), :])
        raise RuntimeError("key_cols must contain unique keys")
    return df


def data_frame_from_named_range(xlsx_file, range_name, header=True):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_file, data_only=True, read_only=True)
    full_range = wb.get_named_range(range_name)
    if full_range is None:
        raise ValueError(f'Range "{range_name}" not found in workbook "{xlsx_file}"')

    destinations = list(full_range.destinations)
    if len(destinations) > 1:
        raise ValueError(f'Range "{range_name}" in workbook "{xlsx_file}" contains more than one region.')

    sheet_name, region_name = destinations[0]
    ws = wb[sheet_name]
    region = ws[region_name]
    df = pd.DataFrame([cell.value for cell in row] for row in region)

    if header:
        df.columns = df.iloc[0, :]
        df.drop(index=0, axis=1, inplace=True)
        df = df.infer_objects()

    return df


def remove_rows_with_invalid_keys(df: pd.DataFrame, keys: list, index_level=None) -> pd.DataFrame:
    if index_level is None:
        if df.loc[~df.index.isin(keys), :].shape[0] > 0:
            print(f"WARNING: the following rows are ignored because the index "
                  f"is not in the list of allowed keys:\n"
                  f"{df.loc[~df.index.isin(keys), :]}")
            return df.loc[df.index.isin(keys), :]
        else:
            return df
    else:
        if df.loc[~df.index.get_level_values(index_level).isin(keys), :].shape[0] > 0:
            print(f"WARNING: the following rows are ignored because the {index_level} "
                  f"is not in the list of allowed keys:\n"
                  f"{df.loc[~df.index.get_level_values(index_level).isin(keys), :]}")
            return df.loc[df.index.get_level_values(index_level).isin(keys), :]
        else:
            return df


def check_referencial_integrity(column: pd.Series, keys):
    """
    Check if each value in the column is from the set of keys.
    If not throw an exception.

    Args:
        column: column with foreign keys
        keys: list / set of primary keys

    Returns:
        NoneType

    """
    if (~column.isin(keys)).any():
        print(column.loc[~column.isin(keys)])
        raise RuntimeError(f"Referential integrity violation")


def check_if_keys_exist_exactly_once(column: pd.Series, keys):
    """ Check if there is exactly one row for each key.
    If not throw an exception.
    There may be other rows where the index is not in keys."""
    # Check if all keys exist
    if len(set(keys) - set(column.keys())) > 0:
        raise RuntimeError("Missing keys")
    # Check if keys are unique
    tmp = column[keys]
    if len(tmp.keys().unique()) < len(tmp):
        raise RuntimeError("Duplicate keys")
