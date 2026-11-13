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
- ELBOW_ROM_OLD_TO_NEW.txt   old -> new code lines for unwrapping and quaternion elbow
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

It prepares the data, calculates the features, writes the outputs
and reads them back to check every value. OUTPUT-run-01 gets one folder
per kind of recording:

    Features_Xsens_Task_1-4/Pazienti/ID/Xsens_tasks/Xsens_tasks.xlsx
        tasks 1-4, one row per repetition; complete visits only
        (32 rows: 4 tasks x 2 arms x 4 repetitions, every feature filled)
    Features_Xsens_Task_1-4/Pazienti_Paretic_AVG/ID/Xsens_tasks/Xsens_tasks.xlsx
        per complete visit and task: mean of the 4 paretic repetitions
    features b&b/id_b&b.csv
        Box and Blocks, paretic arm, one row per visit
    features XSens Motore/id.csv
        Xsens on MOTORE, paretic arm, one row per visit and task
        (task 1 = mot1, task 2 = mot2)

The two CSVs have the columns subject, task, timepoint, arto, arto_eff,
label, n_samples, duration_s, the 15 features, n_degenerate and
degenerate_features. arto = arm from the clinical workbook, arto_eff = arm
used; they differ only for a patient with a side override (local table
predictor_import/side_overrides.json). Values in full precision.

Next to it the same run writes OUTPUT-run-01_SPEED_LENGTH: the same files,
rows and columns, but the four vel_mean_* divided by the length of the lever
that moves each point (unit: lever lengths per second). The other 11
features are identical. This is the speed used in the "length" reports and
the IEEE paper. See section 4.

Run details stay outside OUTPUT:

    derived/xsens/<run>/run.json            files, incomplete visits, rules
    derived/xsens/<run>/runs/PAZIENTI/SUBJECT_ID/xsens/features.csv   both arms, all visits

- More patients: repeat --subject.
- Existing outputs are never overwritten: use a new output folder.
- Features only, no Excel (data already prepared):

    predictor_import/.venv/bin/python -m xsens_code xsens-features \
      standardized/PAZIENTI/SUBJECT_ID --output derived/xsens/features-01


4. WHAT IS CALCULATED (15 Xsens features per repetition)

Every Xsens recording goes through this code; no other feature code is used.

- 4 mean speeds      trunk, shoulder, upper arm, forearm, in m/s
                     (filter x, y, z velocity first, then take the speed)
- 4 SPARC values     smoothness of the same four speeds
- 3 shoulder ROMs    max - min of filtered, unwrapped Euler angles
- 1 elbow ROM        rom_elbow_bending: angle between upper arm and forearm,
                     from quaternions. Unsigned: it cannot tell flexion
                     from hyperextension.
- 2 trunk ranges     max - min of filtered, unwrapped ergonomic angles
- 1 hand DTW         distance between the two hands' trajectories

Which part of the recording:
- tasks 1-4          five markers = four repetitions, one row per repetition.
                     Any other marker count: recording skipped, never repaired.
- B&B, mot1, mot2    no repetitions: one row for the whole recording.
                     Markers are not read or checked. Empty edges (missing
                     samples at the start or end) are left out; gaps inside
                     are never filled. Hand DTW compares the whole recording
                     of the two arms (same visit, same activity).
Method ID: xsens-15-features-quaternion-elbow-finite-span-vector-speed-v3

Length-normalized speeds (the _SPEED_LENGTH folder):
- each speed is divided by the model length of its lever, one number per
  recording, read from the Segment Position sheet (median over the frames):
      forearm (elbow point)       / upper arm (Upper Arm to Forearm origin)
      upper arm (shoulder joint)  / Pelvis origin to Upper Arm origin
      shoulder                    / Pelvis origin to Shoulder origin
      trunk (T8)                  / Pelvis origin to T8 origin
- why: the body model's dimensions are entered differently in Bari,
  Bicocca and Foggia, and some Foggia patients' model changes between visits
- dividing the mean speed is the same as dividing every sample (filter,
  speed and mean all scale with a constant); only float rounding differs
- features.csv keeps m/s and gets the four lengths (lever_*_m columns);
  SPARC does not change with a constant scale, so it stays as it is
- a missing lever leaves only that normalized speed blank, with a warning

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
- feature_importance_code/ which features separate paretic from non-paretic
                           arms, and healthy from simulated movement
- correlation_code/        movement and robot measures vs Fugl-Meyer and
                           Box and Blocks, between and within patients
- schemas/                 contracts for the prepared data
- archive/xsens_code/src/  three old modules, used only as test references

Local only (not in this repository): Garmin raw explorer, plausibility
screening, Drive tools, old methods, report renderers.


6. TESTS

    predictor_import/.venv/bin/python -m pip install -r requirements-dev.txt
    predictor_import/.venv/bin/python -m pytest -q

The two analysis folders have their own pinned environment (CLI_GUIDE.txt
section 12):

    .venv-analysis/bin/python -m pytest correlation_code/tests feature_importance_code/tests

Tests use synthetic data only.
Passing tests check the code, not clinical validity.


7. LICENCE

No project-wide licence yet.
SPARC credit: THIRD_PARTY_NOTICES.md and xsens_code/LICENSE-SPARC.txt.
