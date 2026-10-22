GARMIN CODE

Turns the prepared Fitrockr / Garmin exports into features per assessment
T1-T10: HRV from raw beats, steps, MET, calories, respiration, stress,
resting heart rate and sleep, in four 6-hour blocks per day.

- commands               ../CLI_GUIDE.txt, section 5
- features and code lines ../CODE_MAP.txt, section 6
- rules and open points  ../docs/GARMIN_ASSESSMENTS.txt


FILES

- garmin_reader.py          reads the export tables, checks hashes, by local date
- garmin.py                 rules per signal (zeros = missing, beat filters),
                            6-hour block values, sleep sessions
- garmin_hrv_notebook.py    DEFAULT HRV: first-wave notebook method
- garmin_hrv.py             other HRV: gap-free pieces (--hrv-method segments)
- assessments.py            T windows, wear and 3-day rules, equal-day means
- assessment_export.py      writes the Excel and checks it
- alignment.py              T0-T10 on clinical days, own monitoring periods
- garmin_workflow.py        runs over chosen dates, provenance
- garmin_matrices.py        optional matrix views
- __main__.py               commands: garmin-assessments, garmin-windows,
                            garmin-clinical, garmin-align, garmin-period-template
- tests/                    synthetic tests, no patient data

Local only (not shared): scripts/ (raw CSV explorer), plausibility.py.
