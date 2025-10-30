import io, os, struct, sys, csv, math, argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Funzione di supporto per rimuovere il padding (ovvero spazio finale) se presente
def strip_padding(b: bytes) -> bytes:
    return b[:-1] if (b and (len(b) % 2 == 1) and b.endswith(b"\x20")) else b # Condizione di padding

def decode_float4_list(b: bytes) -> List[Optional[float]]:
    out: List[Optional[float]] = []               # Inizializza la lista di output
    for i in range(0, len(b), 4):                 # Loop ogni 4 byte
        chunk = b[i:i+4]                          # Estrae il chunk di 4 byte
        if len(chunk) < 4:                        # Verifica se il chunk è incompleto
            break
        MISSING_F32_BYTES = b"\xFF\x7F\xFF\xFF"
        out.append(None if chunk == MISSING_F32_BYTES else struct.unpack("<f", chunk)[0])  # Decodifica il float o None
    return out

# Funzione usata per leggere il tag e la lunghezza
def read_tag_and_length(f: io.BufferedReader):
    hdr = f.read(8)                                # Lettura header 8 bytes
    if len(hdr) < 8:                               # Verifica EOF
        return None
    group, element, length = struct.unpack("<HHI", hdr)  # Struttura 2+2+4 bytes letta in little-endian
    return (group, element), length                      # Restituisce la tag e la lunghezza

def decode_text(b: bytes) -> str:
    return strip_padding(b).decode("latin-1", errors="replace").strip()  # Rimuove padding, decodifica e strip

# Rimodella una lista piatta di valori in una griglia 2D
def reshape(vals: List[Optional[float]], rows: int):
    total = len(vals)
    cols = total // rows
    it = iter(vals)
    grid = [[None] * cols for _ in range(rows)]
    # Dati in ordine meridiano-major → blocchi da rows
    for m in range(cols):
        for r in range(rows):
            raw = next(it, None)
            grid[r][m] = None if raw is None else raw
    return grid, cols

def export_COP_csv_points(
    grid: List[List[Optional[float]]],
    rings: int,
    meridians: int,
    radial_step_mm: float,
    include_missing: bool,
    out_csv: str,
    meta: Optional[Dict[str, str]] = None,
    flip_z: bool = False,
    z_exaggeration: float = 1.0,
    theta_sign: float = +1.0,
    theta_offset_rad: float = 0.0,
    break_on_nan: bool = True
) -> None:
    """Esporta i punti della mappa in un CSV 3D compatibile con LabVIEW."""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)

    with open(out_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        if meta:
            for k, v in meta.items():
                fp.write(f"# {k}: {v}\n")
        fp.write("# units: mm\n")
        w.writerow(["x_mm", "y_mm", "z_mm"])


        phi_step = 2 * math.pi / meridians

        for m in range(meridians):  # ciclo esterno: meridiani
            theta = theta_sign * (m * phi_step) + theta_offset_rad
            c, s = math.cos(theta), math.sin(theta)
            for r_idx in range(rings):  # ciclo interno: anelli
                z = grid[r_idx][m]
                if z is None:
                    # LabVIEW: al primo dato mancante chiude il meridiano
                    if include_missing:
                        fp.write("NaN\tNaN\tNaN\n")
                    if break_on_nan:
                        break
                    else:
                        continue

                r_mm = r_idx * radial_step_mm
                z_val = (-(z or 0.0) if flip_z else (z or 0.0)) * z_exaggeration
                x = r_mm * c
                y = r_mm * s
                fp.write(f"{x:.6f};\t{y:.6f};\t{z_val:.6f} \n")

def export_profile_csv_points(
    grid: List[List[Optional[float]]],
    rings: int,
    meridians: int,
    radial_step_mm: float,
    include_missing: bool,
    out_csv: str,
    flip_z: bool = False,
    z_exaggeration: float = 1.0,
    theta_sign: float = +1.0,
    theta_offset_rad: float = 0.0,
    break_on_nan: bool = True
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)

    with open(out_csv, "w", newline="") as fp:
        w = csv.writer(fp)

        phi_step = 2 * math.pi / meridians

        for m in range(meridians):  # ciclo esterno: meridiani
            theta = theta_sign * (m * phi_step) + theta_offset_rad
            c, s = math.cos(theta), math.sin(theta)
            for r_idx in range(rings):  # ciclo interno: anelli
                z = grid[r_idx][m]
                # print("r:",r_idx)   
                if z is None:
                    # LabVIEW: al primo dato mancante chiude il meridiano
                    if include_missing:
                        fp.write("NaN\tNaN\tNaN\n")
                    if break_on_nan:
                        break
                    else:
                        continue

                r_mm = r_idx * radial_step_mm
                z_val = (-(z or 0.0) if flip_z else (z or 0.0)) * z_exaggeration
                x = r_mm * c
                if c == 1 or c == -1:
                    fp.write(f"{x:.6f};\t{z_val:.6f} \n")