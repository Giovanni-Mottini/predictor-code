"""
Step 3 - Spectral Arc Length (SPARC)

FONTE:
La funzione "sparc()" qui sotto e' una riscrittura dell'implementazione 
ufficiale di Sivakumar Balasubramanian:
    Repository : https://github.com/siva82kb/SPARC
    File       : scripts/smoothness.py
    Funzione   : sparc()

Il codice e' riportato separato dal resto della pipeline per mantenere 
la provenienza: tutto cio' che e' in questo file e' codice di terzi citato.
La pipeline lo usa in src/features.py.
"""
from __future__ import annotations
import numpy as np

def sparc(movement, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """
    Calcola la fluidità (smoothness) di un segnale di velocita' tramite
    la metrica Spectral Arc Length (SPARC).

    Parametri
    -
    movement : np.array
        Segnale di velocita' di cui calcolare la fluidità.
    fs : float
        Frequenza di campionamento.
    padlevel : int, opzionale
        Livello di zero-padding per la FFT (default 4).
    fc : float, opzionale
        Frequenza di taglio massima per il calcolo dell'arc length
        (default 10.0 Hz). E' il limite di integrazione dello spettro.
    amp_th : float, opzionale
        Soglia di ampiezza dello spettro normalizzato per la selezione
        della banda rilevante (default 0.05).

    Ritorna
    -
    sal : float
        Valore SPARC (sempre <= 0; più vicino a 0 = più fluido).
    (f, Mf) : tuple
        Spettro di Fourier normalizzato completo (frequenze, magnitudo).
    (f_sel, Mf_sel) : tuple
        Porzione di spettro effettivamente usata per l'arc length.
    """
    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement)) + padlevel)))

    # Frequency
    f = np.arange(0, fs, fs / nfft)
    # Normalized magnitude spectrum
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf / max(Mf)

    # Indices to choose only the spectrum within the given cut off frequency Fc.
    # NOTE: This is a low pass filtering operation to get rid of high frequency
    # noise from affecting the next step (amplitude threshold based cut off for
    # arc length calculation).
    fc_inx = ((f <= fc) * 1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Choose the amplitude threshold based cut off frequency.
    # Index of the last point on the magnitude spectrum that is greater than
    # or equal to the amplitude threshold.
    inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate arc length
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    return new_sal, (f, Mf), (f_sel, Mf_sel)