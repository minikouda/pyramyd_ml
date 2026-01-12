from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, List, Tuple

import pandas as pd


def load_data(filepath):
    return pd.read_csv(filepath)


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    s = str(x).strip().lower()
    return s in {"", "na", "n/a", "none", "null", "nan"}


def to_float(x: Any) -> float:
    """
    Notebook-compatible float conversion:
    - returns NaN for missing/unparseable values
    - strips '%' and commas
    """
    if _is_missing(x):
        return math.nan
    s = str(x).strip().replace("%", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return math.nan


def parse_listish(x: Any) -> List[str]:
    """
    Notebook-compatible list parsing:
    - missing -> []
    - list/tuple/set -> cleaned strings
    - JSON list strings -> parsed
    - Python-literal list strings -> parsed (ast.literal_eval)
    - delimited strings (; | ,) -> split
    - fallback -> [s]
    """
    if _is_missing(x):
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(t).strip() for t in x if str(t).strip()]

    s = str(x).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        # Try JSON first (matches old notebook behavior)
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(t).strip() for t in obj if str(t).strip()]
        except Exception:
            pass
        # Then try python literal (handles single quotes)
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple, set)):
                return [str(t).strip() for t in obj if str(t).strip()]
        except Exception:
            pass

    for sep in [";", "|", ","]:
        if sep in s:
            return [t.strip() for t in s.split(sep) if t.strip()]

    return [s]


def parse_salary(x: Any) -> Tuple[float, float, float]:
    """
    Notebook-compatible salary parsing:
    Returns (min, max, median) from strings like "$90k-$120k".
    Missing/unparseable -> (NaN, NaN, NaN)
    """
    if _is_missing(x):
        return (math.nan, math.nan, math.nan)

    s = str(x).lower()
    s = s.replace("$", "").replace("usd", "").strip()

    # Find numbers with optional k/m suffix (more robust than naive "k"->"000")
    tokens = re.findall(r"(\d+(?:\.\d+)?)([km]?)", s)
    nums: List[float] = []
    for n_str, suf in tokens:
        try:
            v = float(n_str.replace(",", ""))
        except Exception:
            continue
        if suf == "k":
            v *= 1_000.0
        elif suf == "m":
            v *= 1_000_000.0
        nums.append(v)

    if not nums:
        # fallback: original notebook behavior (digits/commas only)
        raw_nums = re.findall(r"([0-9][0-9,]*)", s)
        try:
            nums = [float(n.replace(",", "")) for n in raw_nums]
        except Exception:
            nums = []

    if len(nums) == 0:
        return (math.nan, math.nan, math.nan)
    if len(nums) == 1:
        return (nums[0], nums[0], nums[0])

    mn, mx = min(nums), max(nums)
    return (mn, mx, (mn + mx) / 2.0)
