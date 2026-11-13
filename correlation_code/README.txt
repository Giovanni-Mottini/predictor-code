CORRELATION_CODE - MOVEMENT MEASURES AND CLINICAL SCALES

Correlates movement measures of the paretic arm with Fugl-Meyer (A-D) and
Box and Blocks (paretic hand), at two levels:

- between   Spearman correlation of patient means (at least 4 patients)
- within    repeated-measures correlation (rmcorr): change inside each patient;
            p from 5000 within-patient shuffles of the visits, plus the t-test p,
            a 95% CI and the range when one patient is left out
Holm and Benjamini-Hochberg over all features x scales of one table, per level.


THREE STUDIES (one command each)

- tasks          Xsens tasks 1-4: per visit and task the mean of the repetitions
                 (at least 2), z-scored per task, then the mean over tasks (at
                 least 2)
- xsens-motore   Xsens during the robot sessions: mot1 = Evaluation, mot2 =
                 TraIns; one whole recording per visit
- robot          the robot's own scores (MOTORE fgXX_export CSVs) of the session
                 on the Xsens day of each visit

Pairing with the scales: workbook T0 -> visit T1; T5, T6, ... -> the same
label. Weekly visits T1-T4 have no full scales.


HOW TO RUN

From the project root, in an environment made from requirements.txt:

    python -m correlation_code tasks        --features <run> --workbooks <dir> --speed length --output <new>
    python -m correlation_code xsens-motore --features <run> --workbooks <dir> --speed length --output <new>
    python -m correlation_code robot        --features <run> --workbooks <dir> --exports <dir> [<dir> ...] --output <new>

- --features    xsens_code run folder(s) or features.csv
- --workbooks   folder with <ID>/<clinical workbook>.xlsx (one per patient); only
                the visit labels, their dates and the scale rows are read
- --exports     a patients folder (<ID>/MOTORE/*_export/...) and/or single
                fgXX_export folders
- --order       optional importance.csv of feature_importance_code: table rows
                in its paretic-vs-non-paretic Gini order
- --speed       as in feature_importance_code (length, already-normalized, mps)

The output folder must be new. Results hold patient codes and scores: keep
them under reports/ (git-ignored).


OUTPUT (per table: tasks, mot1, mot2, robot_evaluation, robot_trains)

- <table>_visits.csv     the paired visits: scales and values
- <table>_results.csv    one row per scale and feature, both levels
- <table>_table.tex      LaTeX table
- robot_*_pairing.csv    which session each visit used, or why none
- scales.csv, summary.txt, run.json (inputs with SHA-256, package versions)


TESTS

    python -m pytest correlation_code/tests

Synthetic data with made-up codes and dates only.
