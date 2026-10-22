PREDICTOR_IMPORT - PREPARING THE DATA

Finds the source files, reads the clinical metadata and converts Xsens and
Garmin files into verified Parquet + JSON.
The original files are only read, never changed.

    predictor_import/.venv/bin/python -m predictor_import standardize \
      "Dati PREDICTOR" --subject PAZIENTI/FG91 \
      --output standardized --reports derived/xsens/my-preparation

Commands: inventory, adapt, format-xsens, format-garmin,
migrate-xsens-layout, migrate-garmin-layout, standardize.
Details: ../CLI_GUIDE.txt, sections 3 and 11.


HOW IT WORKS

- standardize converts Xsens and Garmin (choose one with --modality)
- clinical workbooks give the paretic side and the exact clinical dates
- conflicts (timepoint, identity, metadata) are flagged in policy.json
  and left out; nothing is guessed or corrected
- every converted table is checked against the source
- a rerun reuses verified, unchanged work
- --reports gets the inventory, outcomes, coverage and source receipts
  (machine-readable files, no Markdown or HTML report)

Local only: Drive refresh (refresh-drive) and the benchmark.
