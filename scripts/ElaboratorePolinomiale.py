import re
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
from mialibreria import *

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Esegue un fit polinomiale robusto su dati CSV.")
    parser.add_argument("input_csv", help="Percorso del file CSV di input.")
    parser.add_argument(
        "-o", "--output_dir",
        help="Directory o percorso base per i file di output (default: stessa directory del CSV di input).",
        default=None
    )
    parser.add_argument("--xcol", help="Nome della colonna x (opzionale).", default=None)
    parser.add_argument("--ycol", help="Nome della colonna y (opzionale).", default=None)
    parser.add_argument("--maxdeg", type=int, default=8, help="Grado massimo da testare (default: 8).")
    parser.add_argument("--sigma", type=float, default=3.0, help="Soglia sigma-clipping (default: 3.0).")

    args = parser.parse_args()

    coeff, deg, keep, (out_clean, out_expanded) = run_pipeline(
        args.input_csv,
        x_col=args.xcol,
        y_col=args.ycol,
        max_deg=args.maxdeg,
        sigma=args.sigma,
    )

    # Se l'utente ha specificato una directory di output, sposta i file
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in [out_clean, out_expanded]:
            new_path = out_dir / Path(f).name
            Path(f).replace(new_path)
        print(f"\nFile spostati nella directory di output: {out_dir}")

