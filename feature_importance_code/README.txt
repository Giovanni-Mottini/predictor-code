FEATURE_IMPORTANCE_CODE - WHICH XSENS FEATURES MATTER

Ranks the 14 per-arm Xsens features (the 15 xsens_code features without hand
DTW, which compares two arms) for two comparisons:

- paretic_vs_non_paretic   patients: which arm made a repetition
- healthy_vs_simulated     volunteers: healthy repetition or simulated paresis

Two measures, both on people the model has not seen:
- Gini            impurity importance of a Random Forest (100 trees, seed 42)
- held-out drop   loss of balanced accuracy on the test people when the
                  feature is shuffled 20 times within person, visit and task
Patients: leave one patient out, each patient weighted once.
Volunteers: 5 folds grouped by person, balanced class weights.
Details: importance.py.


HOW TO RUN

From the project root, in an environment made from requirements.txt:

    python -m feature_importance_code run \
      --patients <pipeline work directory>/run \
      --speed length --output reports/feature_importance/<new folder>

    python -m feature_importance_code run \
      --volunteers <features.csv or folder> [<more> ...] \
      --speed already-normalized --output reports/feature_importance/<new folder>

--patients and --volunteers take a features.csv, a folder holding one, or an
xsens_code run folder (runs/<cohort>/<ID>/xsens/features.csv).

--speed
- length               m/s divided by each recording's lever length (needs the
                       lever_*_m columns of the current xsens_code)
- already-normalized   the tables were divided upstream
- mps                  m/s as stored
One --speed per run: run the two comparisons separately if their tables
store speeds differently.

The output folder must be new. Results name people: keep them under reports/
(git-ignored).


OUTPUT

- importance.csv            per comparison and feature: Gini, held-out drop, ranks
- importance_by_fold.csv    the same per fold
- folds.csv                 people and balanced accuracy of each fold
- table_ranks.tex           LaTeX rank table (bold = top three, dagger = held-out
                            drop <= 0)
- importance_gini.png, importance_held_out.png
- run.json                  command, inputs with SHA-256, counts, package versions


TESTS

    python -m pytest feature_importance_code/tests

Synthetic tables with made-up codes only (tests/synthetic.py).
