import re
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List

# --------- Lettura CSV e coercizione numerica ---------
def robust_coerce_numeric(s: pd.Series) -> pd.Series:
    def to_num(v):
        if pd.isna(v): return np.nan
        t = str(v).strip()
        if t == "": return np.nan
        # normalizza decimali/sep migliaia
        if ',' in t and '.' in t and t.rfind(',') > t.rfind('.'):
            t = t.replace('.', '').replace(',', '.')
        elif ',' in t and '.' not in t:
            t = t.replace('.', '').replace(',', '.')
        else:
            t = t.replace("'", "").replace(" ", "")
        t = re.sub(r"[^0-9eE+\-\.]", "", t)
        try: return float(t)
        except: return np.nan
    return s.apply(to_num)

def smart_read_csv(path: Path) -> pd.DataFrame:
    tries = [
        dict(sep=None, engine="python"),
        dict(sep=";", engine="python"),
        dict(sep=",", engine="python"),
        dict(sep="\t", engine="python"),
        dict(sep=r"\s+", engine="python"),
    ]
    for kw in tries:
        try:
            return pd.read_csv(path, **kw)
        except Exception:
            continue
    # ultimo tentativo senza header
    for kw in tries:
        try:
            return pd.read_csv(path, header=None, **kw)
        except Exception:
            continue
    raise RuntimeError("Impossibile leggere il CSV.")

def pick_numeric_xy(df: pd.DataFrame, x_col: Optional[str]=None, y_col: Optional[str]=None) -> Tuple[pd.Series, pd.Series, List[str]]:
    if x_col and y_col:
        x = robust_coerce_numeric(df[x_col])
        y = robust_coerce_numeric(df[y_col])
        mask = x.notna() & y.notna()
        return x[mask].reset_index(drop=True), y[mask].reset_index(drop=True), [x_col, y_col]

    # altrimenti scelgo automaticamente le due migliori colonne numeriche
    coerced = {c: robust_coerce_numeric(df[c]) for c in df.columns}
    scores = sorted([(coerced[c].notna().mean(), c) for c in df.columns], reverse=True)
    if len(scores) < 2:
        raise ValueError("Meno di due colonne utili.")
    c1, c2 = scores[0][1], scores[1][1]
    x = coerced[c1]
    y = coerced[c2]
    m = x.notna() & y.notna()
    return x[m].reset_index(drop=True), y[m].reset_index(drop=True), [c1, c2]

# --------- Selezione grado con AICc + fit robusto (sigma-clipping) ---------
def polyfit_aicc(x: np.ndarray, y: np.ndarray, deg: int):
    coeff = np.polyfit(x, y, deg)
    yhat = np.polyval(coeff, x)
    resid = y - yhat
    n = len(y); k = deg + 1
    rss = float(np.sum(resid**2))
    if n <= k + 1:
        return coeff, rss, np.inf
    aic = n * np.log(rss / n) + 2 * k
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    return coeff, rss, aicc

def select_degree(x: np.ndarray, y: np.ndarray, max_deg: int = 8) -> int:
    best_deg, best_aicc = None, np.inf
    for d in range(1, min(max_deg, len(y)-1) + 1):
        try:
            _, _, aicc = polyfit_aicc(x, y, d)
            if aicc < best_aicc:
                best_aicc, best_deg = aicc, d
        except np.linalg.LinAlgError:
            pass
    return best_deg or 1

def robust_polyfit(
    x: np.ndarray, y: np.ndarray,
    init_deg: Optional[int] = None, max_deg: int = 8,
    sigma: float = 3.0, max_iters: int = 10, max_removal_frac: float = 0.3
):
    deg = init_deg or select_degree(x, y, max_deg=max_deg)
    keep = np.ones_like(y, dtype=bool)
    history = []
    for it in range(max_iters):
        coeff = np.polyfit(x[keep], y[keep], deg)
        resid = y[keep] - np.polyval(coeff, x[keep])
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        scale = 1.4826 * mad if mad > 0 else (np.std(resid) if np.std(resid) > 0 else 1e-9)
        thr = sigma * scale
        kept_idx = np.where(keep)[0]
        out_loc = np.where(np.abs(resid - med) > thr)[0]
        out_glob = kept_idx[out_loc]
        # non togliere troppi punti in un colpo
        max_remove = int(max_removal_frac * len(y))
        out_glob = out_glob[:max_remove] if len(out_glob) > max_remove else out_glob
        history.append(dict(iter=it+1, removed=len(out_glob), scale=scale))
        if len(out_glob) == 0: break
        keep[out_glob] = False

    coeff = np.polyfit(x[keep], y[keep], deg)
    rmse = float(np.sqrt(np.mean((y[keep] - np.polyval(coeff, x[keep]))**2)))
    return coeff, keep, deg, rmse, history

# --------- Pipeline principale ---------
def run_pipeline(
    csv_path: str,
    x_col: Optional[str] = None, y_col: Optional[str] = None,
    max_deg: int = 8, sigma: float = 3.0
):
    df = smart_read_csv(Path(csv_path))
    x, y, used = pick_numeric_xy(df, x_col, y_col)
    xv, yv = x.to_numpy().astype(float), y.to_numpy().astype(float)

    coeff, keep, deg, rmse, hist = robust_polyfit(xv, yv, init_deg=None, max_deg=max_deg, sigma=sigma)

    # Interpolazione densa nello stesso range
    x_dense = np.linspace(float(xv.min()), float(xv.max()), 500)
    y_dense = np.polyval(coeff, x_dense)

    # Salvataggi
    clean_df = pd.DataFrame({"x": xv[keep], "y": yv[keep]})
    expanded_df = pd.DataFrame({"x": x_dense, "y_fit": y_dense})
    out_clean = Path(csv_path).with_name(Path(csv_path).stem + "_cleaned.csv")
    out_expanded = Path(csv_path).with_name(Path(csv_path).stem + "_expanded.csv")
    clean_df.to_csv(out_clean, index=False)
    expanded_df.to_csv(out_expanded, index=False)

    # Log sintetico
    print("=== Riepilogo ===")
    print("Colonne usate:", used)
    print("Punti originali:", len(xv))
    print("Outlier rimossi:", int((~keep).sum()))
    print("Grado selezionato:", deg)
    print("RMSE (dati puliti):", rmse)
    print("Coefficienti (da x^{} a costante):".format(deg), coeff)
    print("Salvati:", out_clean, "e", out_expanded)

    # Grafici veloci
    plt.figure()
    plt.scatter(xv, yv, label="Originali")
    plt.scatter(xv[keep], yv[keep], label="Usati (no outlier)")
    plt.legend(); plt.title("Originali vs puliti"); plt.xlabel("x"); plt.ylabel("y")
    plt.show()

    plt.figure()
    plt.scatter(xv[keep], yv[keep], label="Dati usati")
    plt.plot(x_dense, y_dense, label=f"Polinomio grado {deg}")
    plt.legend(); plt.title("Fit polinomiale e interpolazione"); plt.xlabel("x"); plt.ylabel("y")
    plt.show()

    return coeff, deg, keep, (out_clean, out_expanded)

if __name__ == "__main__":
    # Esempio d'uso: cambia 'cristo.csv' col tuo percorso
    run_pipeline("Profile3.csv", x_col=None, y_col=None, max_deg=8, sigma=3.0)
