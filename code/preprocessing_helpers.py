from __future__ import annotations

import pandas as pd


def fill_missing_string(values):
    return pd.DataFrame(values).fillna("__MISSING__").astype(str)
