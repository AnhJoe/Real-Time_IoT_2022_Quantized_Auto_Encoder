from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_raw(save: bool = True, fmt: str = "parquet"):
    """
    Fetch RT-IoT 2022 from UCI, optionally cache to data/raw, and return X, y, df.

    Parameters
    ----------
    save : bool
        Cache locally under data/raw.
    fmt : {"parquet","csv"}
        Storage format. "parquet" requires pyarrow or fastparquet.
        If parquet isn't available, it will fall back to csv automatically.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    df : pd.DataFrame
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"

    parquet_path = RAW_DIR / "rt_iot2022.parquet"
    csv_path = RAW_DIR / "rt_iot2022.csv"

    # Load from cache if present
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        return df.drop(columns=["target"]), df["target"], df

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df.drop(columns=["target"]), df["target"], df

    # Fetch from UCI
    dataset = fetch_ucirepo(id=942)
    X = dataset.data.features
    y = dataset.data.targets

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    df = X.copy()
    df["target"] = y

    if save:
        if fmt == "parquet":
            try:
                df.to_parquet(parquet_path, index=False)
            except ImportError:
                # No parquet engine installed -> fall back to CSV
                df.to_csv(csv_path, index=False)
        elif fmt == "csv":
            df.to_csv(csv_path, index=False)
        else:
            raise ValueError("fmt must be 'parquet' or 'csv'")

    return X, y, df