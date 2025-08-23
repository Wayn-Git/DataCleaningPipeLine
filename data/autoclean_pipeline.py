"""
AutoClean+: Lightweight automated data cleaning & augmentation
Usage:
    import pandas as pd
    from autoclean_pipeline import AutoClean

    df = pd.read_csv("your_dirty.csv")
    auto = AutoClean(categorical_cols=None, text_cols=None, target_col="Grade")
    result = auto.run(df, gen_synthetic=True, n_synth=100, balance_on="Grade")
    df_cleaned = result["cleaned"]
    df_aug = result["augmented"]
    df_cleaned.to_csv("cleaned.csv", index=False)
    df_aug.to_csv("cleaned_aug.csv", index=False)
"""
from typing import Dict, List, Optional, Tuple
import random
import numpy as np
import pandas as pd
from difflib import get_close_matches
from collections import Counter

class AutoClean:
    def __init__(self,
                 categorical_cols: Optional[List[str]]=None,
                 text_cols: Optional[List[str]]=None,
                 id_cols: Optional[List[str]]=None,
                 target_col: Optional[str]=None,
                 typo_match_cutoff: float=0.8,
                 max_ref_values: int=50):
        self.categorical_cols = categorical_cols
        self.text_cols = text_cols
        self.id_cols = id_cols
        self.target_col = target_col
        self.typo_match_cutoff = typo_match_cutoff
        self.max_ref_values = max_ref_values
        
        self.ref_values: Dict[str, List[str]] = {}
        self.num_stats: Dict[str, Tuple[float, float]] = {}
        self.mode_values: Dict[str, object] = {}
        self.gender_map = {
            "m":"Male","male":"Male","M":"Male",
            "f":"Female","female":"Female","F":"Female"
        }
    
    def _infer_columns(self, df: pd.DataFrame):
        if self.categorical_cols is None:
            self.categorical_cols = [c for c in df.columns if df[c].dtype == 'object']
        if self.text_cols is None:
            self.text_cols = [c for c in self.categorical_cols if df[c].astype(str).str.len().median() > 8]

    def fit(self, df: pd.DataFrame):
        # infer cols if not provided
        self._infer_columns(df)
        # ref values for categorical
        for col in self.categorical_cols:
            vals = df[col].dropna().astype(str).tolist()
            norm_vals = [v.strip() for v in vals]
            most_common = [v for v,_ in Counter(norm_vals).most_common(self.max_ref_values)]
            self.ref_values[col] = most_common
        # numeric stats
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                if len(series):
                    std = series.std(ddof=0) if series.std(ddof=0)>0 else 1.0
                    self.num_stats[col] = (series.median(), std)
        # categorical mode for imputation
        for col in self.categorical_cols:
            mode_val = df[col].dropna().mode()
            self.mode_values[col] = mode_val.iloc[0] if not mode_val.empty else None
        return self
    
    def _fix_gender(self, s: pd.Series) -> pd.Series:
        return s.apply(lambda x: self.gender_map.get(x, self.gender_map.get(str(x).lower(), x)) if pd.notna(x) else x)
    
    def _fix_typos_in_col(self, s: pd.Series, ref_list: List[str]) -> pd.Series:
        cleaned = []
        lower_ref = {v.lower(): v for v in ref_list}
        for val in s.fillna(""):
            if val == "":
                cleaned.append(np.nan)
                continue
            v = str(val).strip()
            if v.lower() in lower_ref:
                cleaned.append(lower_ref[v.lower()])
                continue
            matches = get_close_matches(v, ref_list, n=1, cutoff=self.typo_match_cutoff)
            if matches:
                cleaned.append(matches[0])
            else:
                cleaned.append(v.title())
        return pd.Series(cleaned, index=s.index)
    
    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # numeric
        for col, (med, _) in self.num_stats.items():
            out[col] = out[col].fillna(med)
        # categorical
        for col, mode_val in self.mode_values.items():
            out[col] = out[col].fillna(mode_val)
        return out
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "Gender" in out.columns:
            out["Gender"] = self._fix_gender(out["Gender"])
        for col in self.categorical_cols:
            ref = self.ref_values.get(col, [])
            if not ref:
                continue
            out[col] = self._fix_typos_in_col(out[col].astype(str), ref)
        out = out.drop_duplicates()
        out = self._impute(out)
        return out
    
    def augment_random(self, df: pd.DataFrame, n_rows: int=50, typo_prob: float=0.05) -> pd.DataFrame:
        synth = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                med, std = self.num_stats.get(col, (df[col].median(), df[col].std(ddof=0) or 1.0))
                synth[col] = np.random.normal(med, std, size=n_rows)
                if np.issubdtype(df[col].dtype, np.integer):
                    synth[col] = np.round(synth[col]).astype(int)
            else:
                vals = df[col].dropna().tolist()
                if not vals:
                    synth[col] = [None]*n_rows
                else:
                    synth[col] = np.random.choice(vals, size=n_rows, replace=True)
        synth_df = pd.DataFrame(synth)
        def small_typo(x: str) -> str:
            if not isinstance(x, str) or random.random() > typo_prob or len(x) < 4:
                return x
            i = random.randint(0, len(x)-2)
            return x[:i] + x[i+1] + x[i] + x[i+2:]
        for col in self.text_cols or []:
            synth_df[col] = synth_df[col].apply(lambda x: small_typo(x))
        return synth_df
    
    def balance_classes(self, df: pd.DataFrame, target_col: str, strategy: str="upsample") -> pd.DataFrame:
        if target_col not in df.columns:
            return df.copy()
        out = []
        cls_counts = df[target_col].value_counts(dropna=False)
        if strategy == "upsample":
            max_n = cls_counts.max()
            for cls, n in cls_counts.items():
                subset = df[df[target_col]==cls]
                if n < max_n and len(subset):
                    needed = max_n - n
                    extra = subset.sample(needed, replace=True, random_state=42)
                    out.append(pd.concat([subset, extra]))
                else:
                    out.append(subset)
            return pd.concat(out).sample(frac=1.0, random_state=42).reset_index(drop=True)
        else:
            min_n = cls_counts.min()
            for cls, n in cls_counts.items():
                subset = df[df[target_col]==cls].sample(min_n, replace=False, random_state=42)
                out.append(subset)
            return pd.concat(out).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    def run(self, df: pd.DataFrame, gen_synthetic: bool=True, n_synth: int=50, balance_on: Optional[str]=None):
        self.fit(df)
        cleaned = self.clean(df)
        augmented = cleaned.copy()
        if gen_synthetic:
            synth = self.augment_random(cleaned, n_rows=n_synth)
            augmented = pd.concat([cleaned, synth], ignore_index=True)
        if balance_on:
            augmented = self.balance_classes(augmented, target_col=balance_on, strategy="upsample")
        return {"cleaned": cleaned, "augmented": augmented}
