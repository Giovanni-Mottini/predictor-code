PREDICTOR - code for the Xsens and Garmin data

What it does:
original files  ->  prepared copies  ->  features  ->  Excel workbook per patient

    Dati PREDICTOR/      standardized/        derived/xsens/<run>/    OUTPUT/
    originals            Parquet + metadata   features + checks       patient XLSX

This repository holds only code, schemas, docs and synthetic tests.
Patient data and results are never in Git.


1. WHERE TO READ

- README.txt          this file: install, first run, what is calculated
- CLI_GUIDE.txt       every command, step by step
- CODE_MAP.txt        where each step and feature lives in the code
- docs/GARMIN_*.txt   Garmin rules (assessments, HRV, alignment)
- <folder>/README.txt one short file per code folder


2. INSTALL (once)

Python 3.11 or newer.

    git clone https://github.com/Giovanni-Mottini/predictor-code.git
    cd predictor-code
    python3 -m venv predictor_import/.venv
    predictor_import/.venv/bin/python -m pip install -r requirements.txt

On Windows use predictor_import\.venv\Scripts\python.exe instead.


3. FIRST RUN (one patient, raw files to Excel)

Put the data in "Dati PREDICTOR/" and replace SUBJECT_ID:

    predictor_import/.venv/bin/python -m xsens_code pipeline \
      "Dati PREDICTOR" --subject PAZIENTI/SUBJECT_ID \
      --standardized standardized --output OUTPUT-run-01

It prepares the data, calculates the features, writes the workbook
and reads it back to check every value. You get:

    OUTPUT-run-01/PAZIENTI/SUBJECT_ID/Xsens_tasks/Xsens_tasks.xlsx
    derived/xsens/<run>/run.json
    derived/xsens/<run>/runs/PAZIENTI/SUBJECT_ID/xsens/features.csv

- More patients: repeat --subject.
- Existing workbooks are never overwritten: use a new output folder.
- Features only, no Excel (data already prepared):

    predictor_import/.venv/bin/python -m xsens_code xsens-features \
      standardized/PAZIENTI/SUBJECT_ID --output derived/xsens/features-01


4. WHAT IS CALCULATED (15 Xsens features per repetition)

- 4 mean speeds      trunk, shoulder, upper arm, forearm
                     (filter x, y, z velocity first, then take the speed)
- 4 SPARC values     smoothness of the same four speeds
- 3 shoulder ROMs    max - min of filtered, unwrapped Euler angles
- 1 elbow ROM        rom_elbow_bending: angle between upper arm and forearm,
                     from quaternions. Unsigned: it cannot tell flexion
                     from hyperextension.
- 2 trunk ranges     max - min of filtered, unwrapped ergonomic angles
- 1 hand DTW         distance between the two hands' trajectories

Scope: full recordings of tasks 1-4, five markers = four repetitions.
Method ID: xsens-15-features-quaternion-elbow-finite-span-vector-speed-v3

Garmin features (T1-T10 per patient): see garmin_code/README.txt.

Rules kept everywhere:
- original values, markers and patient identities are never changed
- missing stays missing, always with a reason
- conflicts are flagged, never guessed
- "healthy" in the workbook = the patient's non-paretic arm


5. CODE FOLDERS

- predictor_import/        find sources, read clinical metadata, convert and verify
- xsens_code/              repetitions, features, DTW, Excel export
- garmin_code/             Garmin features per assessment T1-T10
- shared_code/             safe reading and writing, provenance, progress
- schemas/                 contracts for the prepared data
- archive/xsens_code/src/  three old modules, used only as test references

Local only (not in this repository): Garmin raw explorer, plausibility
screening, Drive tools, old methods, report renderers.


6. TESTS

    predictor_import/.venv/bin/python -m pip install -r requirements-dev.txt
    predictor_import/.venv/bin/python -m pytest -q

Tests use synthetic data only.
Passing tests check the code, not clinical validity.


7. LICENCE

No project-wide licence yet.
SPARC credit: THIRD_PARTY_NOTICES.md and xsens_code/LICENSE-SPARC.txt.
