"""
Step 2b - Modulo di preprocessing

Produce segnali pronti per la feature extraction a partire dai dati
grezzi letti da "io_xsens.py". Per ogni file Xsens produce i seguenti 
segnali filtrati e ritagliati nelle 4 finestre di ripetizione:
- velocità scalare dei 4 segmenti chiave (trunk, shoulder, upper_arm,
  forearm), filtrata a 6 Hz e a 15 Hz e ritagliata nelle 4 ripetizioni;
- posizione 3D del polso, filtrata a 6 Hz, ritagliata nelle 4 rep,
  utilizzabile direttamente da librerie DTW;
- 4 angoli articolari per ROM (spalla flex/ext, abd/add, int/ext rot 
  e gomito flex/ext), filtrati a 6 Hz e ritagliati;
- 2 angoli di compensazione del tronco (lateral bending + flex/ext
  rispetto alla verticale), filtrati a 6 Hz e ritagliati.

questo modulo produce vettori ("np.ndarray"). Le feature scalari (medie,
SPARC, ROM, DTW) sono compito di "src/features.py".

Principio di filtraggio: il filtro Butterworth zero-phase ("filtfilt")
viene applicato al segnale intero del file, e solo dopo il segnale viene
spezzato nelle 4 finestre di ripetizione. Questo per evitare transitori 
di bordo all'interno di ogni rep, perché il transitorio resta confinato
ai frame iniziali e finali del file, che sono comunque scartati dal
slicing (i marker delimitano regioni più interne).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend
from .io_xsens import FS_HZ

# Cutoff dei filtri Butterworth (Hz)
CUTOFF_LOW = 6.0    # velocità per medie/ROM, posizione mano, angoli, compensazioni
CUTOFF_HIGH = 15.0  # velocità per SPARC

# 4 segmenti chiave per la velocità scalare
SEGMENT_ROLES = {
    "trunk":     "T8",
    "shoulder":  "Shoulder",
    "upper_arm": "Upper Arm",
    "forearm":   "Forearm",
}

# Mapping arto file
SIDE_PREFIX = {"dx": "Right", "sx": "Left"}

# Angoli articolari per ROM (foglio Joint Angles ZXY)
JOINT_ANGLES_SUFFIX = {
    "shoulder_flex_ext":    "Shoulder Flexion/Extension",
    "shoulder_abd_add":     "Shoulder Abduction/Adduction",
    "shoulder_int_ext_rot": "Shoulder Internal/External Rotation",
    "elbow_flex_ext":       "Elbow Flexion/Extension",
}

# Compensazioni del tronco (foglio Ergonomic Joint Angles ZXY)
TRUNK_COMPENSATION_COL = {
    "lateral_bending": "Vertical_T8 Lateral Bending",
    "flex_ext":        "Vertical_T8 Flexion/Extension",
}

@dataclass
class PreprocessedFile:
    """
    Contenitore dei segnali pronti per la feature extraction.

    Attributi
    -
    meta : dict
        Metadati del file (subject, task, trial, arto) propagati da
        "parse_filename" e "validate_file".
    arto_eff : str
        Lato anatomico effettivo ('dx' o 'sx'). Per file _dx/_sx
        coincide con "meta['arto']"; per file _sim viene determinato
        esternamente dal file excel.
    windows : list[tuple[int, int]]
        Le 4 finestre [start, end) usate per il ritaglio, propagate
        da "io_xsens.get_repetition_windows". Utili in Step 3 per
        risalire alla durata di ogni rep.
    velocity_6hz, velocity_15hz : dict[str, list[np.ndarray]]
        Velocità scalare (m/s) per ognuno dei 4 segmenti.
    hand_position_6hz : list[np.ndarray]
        Posizione 3D del polso (m).
    joint_angles_6hz : dict[str, list[np.ndarray]]
        Angoli articolari (gradi) per i 4 angoli di spalla e
        gomito.
    trunk_compensation_6hz : dict[str, list[np.ndarray]]
        Angoli del tronco rispetto alla verticale (gradi).
    """
    meta: dict
    arto_eff: str
    windows: list[tuple[int, int]]
    velocity_6hz:           dict[str, list[np.ndarray]] = field(repr=False)
    velocity_15hz:          dict[str, list[np.ndarray]] = field(repr=False)
    hand_position_6hz:      list[np.ndarray]            = field(repr=False)
    joint_angles_6hz:       dict[str, list[np.ndarray]] = field(repr=False)
    trunk_compensation_6hz: dict[str, list[np.ndarray]] = field(repr=False)

    def summary(self) -> str:
        """Riepilogo testuale per debug"""
        rep_lens = [end - start for start, end in self.windows]
        lines = [
            f"PreprocessedFile(",
            f"  subject={self.meta.get('subject')!r}, "
            f"task={self.meta.get('task')}, trial={self.meta.get('trial')}, "
            f"arto_file={self.meta.get('arto')!r}, arto_eff={self.arto_eff!r}",
            f"  windows: {self.windows}",
            f"  durate rep (frames): {rep_lens}",
            f"  velocity_6hz:    {list(self.velocity_6hz)} "
            f"× 4 rep × lunghezza variabile",
            f"  velocity_15hz:   {list(self.velocity_15hz)} × 4 rep",
            f"  hand_position_6hz: 4 rep × (n, 3) "
            f"[rep 1 shape: {self.hand_position_6hz[0].shape}]",
            f"  joint_angles_6hz: {list(self.joint_angles_6hz)} × 4 rep",
            f"  trunk_compensation_6hz: "
            f"{list(self.trunk_compensation_6hz)} × 4 rep",
            f")",
        ]
        return "\n".join(lines)

def get_segment_columns(arto: str, role: str) -> list[str]:
    """
    Restituisce le 3 colonne (x, y, z) di Segment Velocity per il
    segmento richiesto.

    Per "role='trunk'" ritorna "['T8 x', 'T8 y', 'T8 z']" (T8 e' un
    segmento centrale e non ha prefisso laterale). Per gli altri segementi
    ritorna le colonne con il prefisso laterale corrispondente all'arto
    ('Right' / 'Left').

    Parametri
    -
    arto : str
        'dx' o 'sx'. Per file _sim usa l'arto_eff dell'excel.
    role : str
        Una delle chiavi di "SEGMENT_ROLES".
    """
    seg = SEGMENT_ROLES[role]
    base = seg if role == "trunk" else f"{SIDE_PREFIX[arto]} {seg}"
    return [f"{base} x", f"{base} y", f"{base} z"]

def scalar_speed(df_vel: pd.DataFrame, arto: str, role: str) -> np.ndarray:
    """
    Velocità scalare del segmento sul segnale intero.

    "||v|| = sqrt(vx^2 + vy^2 + vz^2)"

    Parametri
    -
    df_vel : pd.DataFrame
        Foglio "segment_velocity"
    arto : str
        'dx' o 'sx'.
    role : str
        Una delle chiavi di "SEGMENT_ROLES".

    Ritorna
    -
    np.ndarray
        Array 1D di lunghezza "len(df_vel)", valori in m/s.
    """
    cols = get_segment_columns(arto, role)
    v = df_vel[cols].values.astype(float)
    return np.sqrt(np.sum(v ** 2, axis=1))

def butter_lowpass(signal: np.ndarray, cutoff: float,
                   fs: float = FS_HZ, order: int = 4) -> np.ndarray:
    """
    Filtro Butterworth passa-basso a fase zero ("filtfilt").

    Parametri
    -
    signal : np.ndarray
        Segnale 1D di input.
    cutoff : float
        Frequenza di taglio in Hz.
    fs : float
        Frequenza di campionamento (default "FS_HZ" = 60 Hz).
    order : int
        Ordine del filtro Butterworth, Default 4.

    Ritorna
    -
    np.ndarray
        Segnale filtrato, stessa shape dell'input.
    """
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    padlen = min(3 * order, len(signal) - 1)
    return filtfilt(b, a, signal, padlen=padlen)

def slice_repetitions(signal: np.ndarray,
                      windows: list[tuple[int, int]]
                      ) -> list[np.ndarray]:
    """
    Spezza il segnale già filtrato nelle finestre delle ripetizioni.


    Parametri
    -
    signal : np.ndarray
        Segnale 1D o 2D.
    windows : list[tuple[int, int]]
        Finestre "start, end)". 

    Ritorna
    -
    list[np.ndarray]
        Una lista, una entry per ogni finestra in "windows".
    """
    return [signal[start:end] for start, end in windows]

def detrend_per_window(signal: np.ndarray,
                       windows: list[tuple[int, int]]) -> np.ndarray:
    """
    Detrending lineare per finestra + concatenazione.

    Per ciascuna finestra in "windows" estrae il segmento corrispondente
    da "signal", applica detrending lineare e concatena i risultati. 
    Serve a evitare uno "step" artificioso al confine fra rep adiacenti
    quando si calcola la PSD del segnale concatenato.

    Parametri
    -
    signal : np.ndarray
        Segnale 1D.
    windows : list[tuple[int, int]]
        Finestre "[start, end)" da concatenare. 

    Ritorna
    -
    np.ndarray
        Concatenazione dei pezzi detrendati.
    """
    pieces: list[np.ndarray] = []
    for start, end in windows:
        s = signal[start:end]
        if len(s) >= 2:
            s = detrend(s, type="linear")
        pieces.append(s)
    return np.concatenate(pieces)

# Workflow
def preprocess_file(sheets: dict[str, pd.DataFrame],
                    meta: dict,
                    windows: list[tuple[int, int]],
                    arto_eff: Optional[str] = None) -> PreprocessedFile:
    """
    Produce un :class: 'PreprocessedFile' a partire dai dati grezzi.

    Procedura per ogni segnale (velocità, posizione mano, angoli,
    compensazioni):

    1. Estrae il segnale grezzo dal foglio 
    2. Applica "butter_lowpass" al segnale intero del file 
    3. Spezza con "slice_repetitions" nelle 4 finestre.

    Per la velocità si producono due versioni filtrate (6 Hz e 15 Hz)
    perché serviranno entrambe in Step 3 (6 Hz per mean velocity e
    ROM, 15 Hz per SPARC).

    Parametri
    -
    sheets : dict[str, pd.DataFrame]
        Dict prodotto da "io_xsens.load_xsens_file".
    meta : dict
        Metadati del file (subject, task, trial, arto).
    windows : list[tuple[int, int]]
        4 finestre prodotte da "io_xsens.get_repetition_windows".
    arto_eff : str, opzionale
        Lato anatomico effettivo ('dx' o 'sx'). Se None, default a
        "meta['arto']" (corretto per file _dx/_sx; per file _sim il
        chiamante deve passarlo esplicitamente leggendo l'excel).

    Ritorna
    -
    PreprocessedFile
        Contenitore con tutti i segnali pronti.
    """
    if arto_eff is None:
        arto_eff = meta["arto"]

    df_vel = sheets["segment_velocity"]
    df_pos = sheets["segment_position"]
    df_ja  = sheets["joint_angles_zxy"]
    df_erg = sheets["ergo_joint_angles_zxy"]

    # Velocità scalare, 2 cutoff, per i 4 segmenti
    velocity_6hz:  dict[str, list[np.ndarray]] = {}
    velocity_15hz: dict[str, list[np.ndarray]] = {}
    for role in SEGMENT_ROLES:
        speed_full = scalar_speed(df_vel, arto_eff, role)
        speed_6  = butter_lowpass(speed_full, CUTOFF_LOW)
        speed_15 = butter_lowpass(speed_full, CUTOFF_HIGH)
        velocity_6hz[role]  = slice_repetitions(speed_6,  windows)
        velocity_15hz[role] = slice_repetitions(speed_15, windows)

    # Posizione 3D del polso
    side = SIDE_PREFIX[arto_eff]
    hand_cols = [f"{side} Hand x", f"{side} Hand y", f"{side} Hand z"]
    hand_xyz = df_pos[hand_cols].values.astype(float)  # (n_total, 3)
    hand_xyz_filt = np.column_stack([
        butter_lowpass(hand_xyz[:, i], CUTOFF_LOW) for i in range(3)
    ])
    hand_position_6hz = slice_repetitions(hand_xyz_filt, windows)

    # Angoli articolari 
    joint_angles_6hz: dict[str, list[np.ndarray]] = {}
    for name, suffix in JOINT_ANGLES_SUFFIX.items():
        col = f"{side} {suffix}"
        angle_full = df_ja[col].values.astype(float)
        angle_filt = butter_lowpass(angle_full, CUTOFF_LOW)
        joint_angles_6hz[name] = slice_repetitions(angle_filt, windows)

    # Compensazioni tronco 
    trunk_compensation_6hz: dict[str, list[np.ndarray]] = {}
    for name, col in TRUNK_COMPENSATION_COL.items():
        comp_full = df_erg[col].values.astype(float)
        comp_filt = butter_lowpass(comp_full, CUTOFF_LOW)
        trunk_compensation_6hz[name] = slice_repetitions(comp_filt, windows)

    return PreprocessedFile(
        meta=meta,
        arto_eff=arto_eff,
        windows=windows,
        velocity_6hz=velocity_6hz,
        velocity_15hz=velocity_15hz,
        hand_position_6hz=hand_position_6hz,
        joint_angles_6hz=joint_angles_6hz,
        trunk_compensation_6hz=trunk_compensation_6hz,
    )