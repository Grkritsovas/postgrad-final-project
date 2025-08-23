import pandas as pd
from sklearn.preprocessing import StandardScaler

class SplitScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.columns = None

    def fit(self, X: pd.DataFrame):
        self.columns = list(X.columns)
        self.scaler.fit(X.values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Ensure column order matches training to prevent silent bugs
        X_reordered = X[self.columns]
        Z = self.scaler.transform(X_reordered.values)
        return pd.DataFrame(Z, index=X.index, columns=self.columns)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)