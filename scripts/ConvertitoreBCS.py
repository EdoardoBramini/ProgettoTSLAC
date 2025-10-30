import io, sys, math, argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from mialibreria import *

# La Tag viene idetificata da due valori seguendo il protocollo DICOM: (group, element)
Tag = Tuple[int, int]

# Inserisco i valori standard delle Tag usate nel file BCS (recuperati dalla documentazione)

# Valori standard delle Tag dei dati di mappa di elevazione
TAG_ELEVATION      = (0x0023, 0x0210)    # Dati di elevazione z (31 anelli)
TAG_EXT_ELEVATION  = (0x0023, 0x0211)    # Dati di elevazione z estesa (38 anelli)
TAG_POST_ELEVATION = (0x0023, 0x0212)    # Superficie posteriore di elevazione (31 anelli)

# Valori standard delle Tag dei dati testuali
TAG_FIRSTNAME = (0x0023, 0x0012)         # Nome del paziente
TAG_LASTNAME  = (0x0023, 0x0011)         # Cognome del paziente
TAG_LATERALITY= (0x0020, 0x0060)         # Lateralità dell'occhio (es. DX/SX)
TAG_INSTRUMENT= (0x0023, 0x0017)         # Strumento usato
TAG_STUDYDATE = (0x0008, 0x0020)         # Data dello studio
TAG_NOTES     = (0x0023, 0x0233)         # Note aggiuntive

# Costanti per la decodifica delle mappe di elevazione
RINGS_STANDARD   = 31                    # Numero di anelli standard
RINGS_EXTENDED   = 38                    # Per il set esteso
RADIAL_STEP_MM   = 0.2                   # Spazio radiale tra anelli in mm
MISSING_F32_BYTES = b"\xFF\x7F\xFF\xFF"  # Codice binario per i float mancanti (NaN)

# Struttura dati per ogni elemento decodificato
@dataclass
class Element:
    tag: Tag        # (group, element)
    length: int     # Lunghezza del payload in byte
    raw: bytes      # Dati grezzi del payload
    value: Any      # Valori decodificati da Python


# Classe principale per il parsing dei file BCS
class BCSParser:
    # Inizializzazione
    def __init__(self):
        self.elems: Dict[Tag, Element] = {}         # Definisce il dizionario degli elementi (tag → Element)

    def parse(self, data: bytes) -> Dict[Tag, Element]:
        f = io.BufferedReader(io.BytesIO(data))     # Lettura dati in memoria come stream binario (più efficiente)
        while True:
            head = read_tag_and_length(f)          # Viene letta la tag e la lunghezza
            if head is None:                        # Condizione EOF
                break
            tag, length = head                      # Estrae tag e lunghezza e li assegna a variabili
            raw = f.read(length)                    # Assegna i byte del payload alla variabile raw

            # Decodifica in base al tipo di tag

            # Mappe di elevazione: decodifica lista di float e rimodella e assegna a val
            if tag in (TAG_ELEVATION, TAG_EXT_ELEVATION, TAG_POST_ELEVATION):
                vals = decode_float4_list(raw)                                           # Decodica lista di float
                rows = RINGS_EXTENDED  if tag == TAG_EXT_ELEVATION else RINGS_STANDARD     # Determina numero di anelli
                grid, cols = reshape(vals, rows=rows+3)                                    # Rimodella in griglia 2D
                val = {"grid": grid, "rings": rows, "meridians": cols}                    # Crea dizionario di output
            # Tag di testo: decodifica come stringhe Latin-1 e assegna a val
            elif tag in (TAG_FIRSTNAME, TAG_LASTNAME, TAG_LATERALITY,
                         TAG_INSTRUMENT, TAG_STUDYDATE, TAG_NOTES):
                val = decode_text(raw)             
            # Tag sconosciute: mantieni i byte grezzi e assegna a val
            else:
                val = raw                           
            # Inserisce l'elemento con le rispettive quattro variabili nel dizionario (avendo decodificato il valore)
            self.elems[tag] = Element(tag, length, raw, val)
        return self.elems
    
    # Controller di accesso sicuro: restituisce il valore dell'elemento o None se mancante (Optional[Any])
    def get(self, tag: Tag) -> Optional[Any]:
        el = self.elems.get(tag)
        return el.value if el else None

# -------------------------------------------------------
def main():
    # Impostazione argomenti da linea di comando
    ap = argparse.ArgumentParser(description="BCS → CSV (3D cornea)")
    ap.add_argument("input", help="Input BCS file")
    ap.add_argument("output_csv", help="Output CSV COP filename")
    ap.add_argument("output2_csv", help="Output CSV profile filename")
    ap.add_argument("--map", choices=["auto","extended","elevation","posterior"], default="auto",
                    help="Map type: auto = prefer Extended > Elevation > Posterior")
    ap.add_argument("--include-missing", action="store_true", help="Include None samples in CSV")
    ap.add_argument("--flip-z", action="store_true", help="Invert sign of Z")
    ap.add_argument("--z-exaggeration", type=float, default=1.0,
                    help="Vertical exaggeration factor")
    # Opzioni per la conversione angolare
    ap.add_argument("--theta-mode", choices=["ccw","cw"], default="ccw",
                    help="Direction of meridians: ccw=counterclockwise, cw=clockwise")
    ap.add_argument("--theta-offset-deg", type=float, default=0.0,
                    help="Angular offset in degrees (e.g. 90)")

    args = ap.parse_args()                                      # Parsing argomenti
    theta_sign = +1.0 if args.theta_mode == "ccw" else -1.0     # Determina il segno angolare
    theta_offset_rad = math.radians(args.theta_offset_deg)      # Converte l'offset angolare in radianti

    # Legge il file BCS
    with open(args.input, "rb") as f:
        data = f.read()

    # Inizializza il parser e analizza i dati
    parser = BCSParser()
    parser.parse(data)

    # Sceglie che tipo di mappa di elevazione usare
    tag = None
    if args.map == "extended" or (args.map == "auto" and parser.get(TAG_EXT_ELEVATION) is not None):
        tag = TAG_EXT_ELEVATION
    elif args.map == "elevation" or (args.map == "auto" and parser.get(TAG_ELEVATION) is not None):
        tag = TAG_ELEVATION
    elif args.map == "posterior" or (args.map == "auto" and parser.get(TAG_POST_ELEVATION) is not None):
        tag = TAG_POST_ELEVATION
    if tag is None:
        sys.stderr.write("Non è presente una elevation map nel file (Extended/Elevation/Posterior).\n")
         #sys.exit(2)

    # Estrae il payload della mappa di elevazione
    payload = parser.get(tag)
    if not isinstance(payload, dict) or "grid" not in payload:                  # Verifica formato payload
        sys.stderr.write("Unexpected format: expected dict with 'grid'.\n")     # Messaggio di errore
        #sys.exit(3)

    # Estrae griglia, anelli e meridiani
    grid = payload["grid"]
    rings = payload["rings"]
    meridians = payload["meridians"]

    # Definisce il dizionario dei metadati da includere nell'intestazione del CSV
    meta = {
        "FirstName": str(parser.get(TAG_FIRSTNAME) or ""),
        "LastName": str(parser.get(TAG_LASTNAME) or ""),
        "EyeLaterality": str(parser.get(TAG_LATERALITY) or ""),
        "Instrument": str(parser.get(TAG_INSTRUMENT) or ""),
        "StudyDate": str(parser.get(TAG_STUDYDATE) or ""),
        "Notes": str(parser.get(TAG_NOTES) or ""),
        "Rings": str(rings),
        "Meridians": str(meridians),
        "RadialStep_mm": str(RADIAL_STEP_MM),
    }

    # Esporta i punti in CSV
    export_COP_csv_points(
        grid, rings, meridians, RADIAL_STEP_MM,
        include_missing=args.include_missing,
        out_csv=args.output_csv, 
        meta=meta,
        flip_z=args.flip_z, 
        z_exaggeration=args.z_exaggeration,
        theta_sign=theta_sign, 
        theta_offset_rad=theta_offset_rad
    )
    print(f"CSV COP written: {args.output_csv}") # Conferma di avvenuta scrittura

     # Esporta i punti in CSV
    export_profile_csv_points(
        grid, rings, meridians, RADIAL_STEP_MM,
        include_missing=args.include_missing,
        out_csv=args.output2_csv, 
        flip_z=args.flip_z, 
        z_exaggeration=args.z_exaggeration,
        theta_sign=theta_sign, 
        theta_offset_rad=theta_offset_rad
    )
    print(f"CSV Profile written: {args.output2_csv}") # Conferma di avvenuta scrittura 2

# Inizio esecuzione script
if __name__ == "__main__":
    main()