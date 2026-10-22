"""
Step 3/5 - Feature extraction

Dato un class:`PreprocessedFile` (prodotto da "src/preprocessing.py"),
calcola le 15 features scalari per ognuna delle 4 ripetizioni.

Le 14 features, raggruppate nelle 4 macroaree richieste sono:
- Velocità (4) : mean velocity dei 4 segmenti (trunk, shoulder,
  upper_arm, forearm), dal campo "velocity_6hz".
- Fluidità (4) : SPARC dei 4 segmenti, dal campo "velocity_15hz".
- ROM (4)       : range (max - min) dei 4 angoli articolari, dal campo
  "joint_angles_6hz".
- Compensazioni (2) : range (max - min) dei 2 angoli del tronco rispetto
  alla verticale, dal campo "trunk_compensation_6hz".
- Controllo motorio (1): distanza DTW fra traiettoria della mano paretica 
  e sana 

Le quattro funzioni base da cui calcoliamo tutte le features sono:
- "mean_velocity"    -> media della velocità scalare.
- "spectral_arc_length" -> fluidità del segnale di velocità.
- "angular_range"    -> range max-min, condiviso da ROM e Compensazioni
- "dtw_distance"     -> distanza DTW fra traiettorie

Output: una lista di 4 dict (uno per ripetizione). Ogni dict contiene i
metadati identificativi della ripetizione + le 15 features.
"""
import warnings
import numpy as np
from dtaidistance import dtw_ndim # type: ignore
from .io_xsens import FS_HZ
from .preprocessing import PreprocessedFile
from .sparc import sparc

# Parametri SPARC (default validati scientificamente da Balasubramanian)
SPARC_PADLEVEL = 4
SPARC_FC = 10.0
SPARC_AMP_TH = 0.05

# Prefissi dei nomi colonna per macroarea (header del CSV finale)
VEL_PREFIX = "vel_mean"
SPARC_PREFIX = "sparc"
ROM_PREFIX = "rom"
COMP_PREFIX = "comp_trunk"

# Funzioni base per calcolo features
def mean_velocity(speed: np.ndarray) -> float:
    """
    Media della velocità scalare nella ripetizione.

    "MV = (1/N) * sum_i v_i"

    dove "v_i" è la velocità istantanea già calcolata da
    "preprocessing.py" e "N" il numero di campioni.

    Parametri
    -
    speed : np.ndarray
        Velocità scalare 1D di una ripetizione (m/s), filtrata a 6 Hz.

    Ritorna
    -
    float
        Velocità media (m/s). "np.nan" se la ripetizione è vuota.
    """
    if len(speed) < 1:
        return float("nan")
    return float(np.mean(speed))

def spectral_arc_length(speed_15hz: np.ndarray, fs: float = FS_HZ) -> float:
    """
    Fluidità del segnale di velocità.

    Riceve in ingresso la velocità già filtrata a 15 Hz
    ("velocity_15hz" del PreprocessedFile). I parametri "fc=10.0",
    "amp_th=0.05", "padlevel=4" sono i default validati
    scientificamente da Balasubramanian e non vanno modificati.

    Un valore SPARC è sempre <= 0; più vicino a 0 = movimento più
    fluido, più negativo = movimento meno fluido.

    Parametri
    -
    speed_15hz : np.ndarray
        Velocità scalare 1D di una ripetizione, filtrata a 15 Hz.
    fs : float
        Frequenza di campionamento (default "FS_HZ" = 60 Hz).

    Ritorna
    -
    float
        Valore SPARC. "np.nan" se la ripetizione è troppo corta per
        un calcolo significativo (< 2 campioni).
    """
    if len(speed_15hz) < 2:
        return float("nan")
    sal, _, _ = sparc(speed_15hz, fs=fs, padlevel=SPARC_PADLEVEL,
                      fc=SPARC_FC, amp_th=SPARC_AMP_TH)
    return float(sal)

def angular_range(angle: np.ndarray) -> float:
    """
    Range articolare (max - min) di un angolo nella ripetizione.

    Condivisa da ROM  e Compensazioni del tronco: in entrambi i casi 
    la feature è l'ampiezza angolare totale nella finestra, cioè 
    la differenza fra il valore massimo e il valore minimo.

    Parametri
    -
    angle : np.ndarray
        Serie 1D di un angolo (gradi) in una ripetizione, filtrata a 6 Hz.

    Ritorna
    -
    float
        Ampiezza angolare (gradi). "np.nan" se la ripetizione è
        vuota.
    """
    if len(angle) < 1:
        return float("nan")
    return float(np.max(angle) - np.min(angle))

def center_trajectory(traj: np.ndarray, baseline_frames: int = 10) -> np.ndarray:
    """
    Centra una traiettoria 3D sottraendo il proprio centroide.

    I dati di posizione di Xsens (foglio Segment Position) sono in un
    sistema di riferimento globale: la mano destra e la sinistra sono 
    su due lati opposti del corpo, separate da un offset costante di 
    circa 35-40 cm. Senza correzione questo offset domina la distanza
    DTW, nascondendo la vera differenza di forma del movimento che 
    vogliamo misurare. Sottraendo a ogni traiettoria la propria posizione
    media, la DTW diventa invariante alla traslazione.

    Parametri
    -
    traj : np.ndarray
        Traiettoria, shape (n, 3).

    Ritorna
    -
    np.ndarray
        Traiettoria centrata, stessa shape.
    """
    if len(traj) == 0:
        return traj
        
    if len(traj) < baseline_frames:
        # Se la traiettoria è troppo corta, usa il primo punto come baseline
        baseline = traj[0]
    else:
        # Calcola il punto di partenza 
        baseline = np.mean(traj[:baseline_frames], axis=0)
        
    return traj - baseline

def dtw_distance(traj_a: np.ndarray, traj_b: np.ndarray,
                 center: bool = True) -> float:
    """
    Distanza DTW (Dynamic Time Warping) fra due traiettorie 3D.

    Misura quanto due traiettorie della mano sono diverse come forma,
    allineandoli nel tempo. Calcolata con la libreria dtaidistance 
    (dtw_ndim).

    Parametri
    -
    traj_a : np.ndarray
        Traiettoria 3D, shape (n, 3) (n istanti, coordinate x,y,z).
    traj_b : np.ndarray
        Traiettoria 3D, shape (m, 3). Puo' avere lunghezza diversa 
        da traj_a.

    Ritorna
    -
    float
        Distanza DTW. np.nan se una delle due traiettorie è vuota.
    """
    if traj_a is None or traj_b is None:
        return float("nan")
    if len(traj_a) < 1 or len(traj_b) < 1:
        return float("nan")
    a = np.ascontiguousarray(traj_a, dtype=float)
    b = np.ascontiguousarray(traj_b, dtype=float)
    if center:
        a = center_trajectory(a)
        b = center_trajectory(b)
    return float(dtw_ndim.distance(a, b))

# Workflow
def extract_features(pre: PreprocessedFile) -> list[dict]:
    """
    Calcola le 14 features scalari (no DTW) per ognuna delle 4 ripetizioni.

    Per ogni ripetizione produce un dict con:
    - metadati identificativi: "subject", "task", "trial",
      "arto" (suffisso file), "arto_eff" (lato anatomico effettivo),
      "rep", "n_samples", "duration_s";
    - le 14 features;
    - "degenerate_features": lista dei nomi feature risultate "nan"
      a causa di una ripetizione degenere (vuota/troppo corta).

    Naming delle 14 colonne:
    - Velocità      : vel_mean_{trunk,shoulder,upper_arm,forearm}
    - Fluidità      : sparc_{trunk,shoulder,upper_arm,forearm}
    - ROM           : rom_{shoulder_flex_ext,shoulder_abd_add,
      shoulder_int_ext_rot,elbow_flex_ext}
    - Compensazioni : comp_trunk_{lateral_bending,flex_ext}

    Parametri
    -
    pre : PreprocessedFile
        Contenitore dei segnali prodotto da "preprocess_file".

    Ritorna
    -
    list[dict]
        4 dict (uno per ripetizione).
    """
    n_rep = len(pre.windows)
    rows: list[dict] = []

    for r in range(n_rep):
        start, end = pre.windows[r]
        n_samples = end - start

        row: dict = {
            "subject": pre.meta.get("subject"),
            "task": pre.meta.get("task"),
            "trial": pre.meta.get("trial"),
            "arto": pre.meta.get("arto"),
            "arto_eff": pre.arto_eff,
            "rep": r + 1,
            "n_samples": n_samples,
            "duration_s": round(n_samples / FS_HZ, 4),
        }
        degenerate: list[str] = []

        # Velocità (4): mean velocity dal segnale 6 Hz
        for role, speed_list in pre.velocity_6hz.items():
            col = f"{VEL_PREFIX}_{role}"
            val = mean_velocity(speed_list[r])
            row[col] = val
            if np.isnan(val):
                degenerate.append(col)

        # Fluidità (4): SPARC dal segnale 15 Hz
        for role, speed_list in pre.velocity_15hz.items():
            col = f"{SPARC_PREFIX}_{role}"
            val = spectral_arc_length(speed_list[r])
            row[col] = val
            if np.isnan(val):
                degenerate.append(col)

        # ROM (4): range dei 4 angoli articolari a 6 Hz
        for name, angle_list in pre.joint_angles_6hz.items():
            col = f"{ROM_PREFIX}_{name}"
            val = angular_range(angle_list[r])
            row[col] = val
            if np.isnan(val):
                degenerate.append(col)

        # Compensazioni (2): range dei 2 angoli del tronco a 6 Hz
        for name, angle_list in pre.trunk_compensation_6hz.items():
            col = f"{COMP_PREFIX}_{name}"
            val = angular_range(angle_list[r])
            row[col] = val
            if np.isnan(val):
                degenerate.append(col)

        row["degenerate_features"] = degenerate
        if degenerate:
            warnings.warn(
                f"Ripetizione NaN: {pre.meta.get('subject')} "
                f"task{pre.meta.get('task')} rep{r + 1} "
                f"(n_samples={n_samples}); feature nan: {degenerate}"
            )

        rows.append(row)

    return rows