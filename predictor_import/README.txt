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

Two local tables hold dataset-owner decisions about real participants, so
they are git-ignored and optional (without them nothing changes):
- side_overrides.json   paretic side read from the arm that performs
- reexports.json        a re-export folder (e.g. XSens_new) replaces the
                        first export of named visits. In those visits:
                        the re-export replaces the recording of the same
                        activity and arm; an original left without one is
                        not used if it is HD (Normal only); a reviewed
                        marker substitution can give a re-export the
                        original export's markers (the exported table stays
                        as markers_as_exported.parquet). See reexports.py.
- a workbook whose shared strings hold a bare '&' (MVN writes file names
  unescaped) is read through an in-memory repair; the descriptor records it
  in source_repairs and the warning source_text_repaired
- --reports gets the inventory, outcomes, coverage and source receipts
  (machine-readable files, no Markdown or HTML report)

Local only: Drive refresh (refresh-drive) and the benchmark.
