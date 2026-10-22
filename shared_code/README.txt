SHARED_CODE - COMMON TOOLS

- reader.py        patient identity, picks records, safe paths, contract errors
- io.py            safe (atomic) JSON writes
- artifacts.py     standard fields, protected output folders, safe CSV/text writes
- provenance.py    SHA-256 of the code, with repository paths
- progress.py      progress messages on stderr

Algorithms are in xsens_code/ and garmin_code/.
Import and conversion are in predictor_import/.
Local only: report renderers, plausibility studies, release tools, compat shims.
