XSENS_CODE - MOVEMENT FEATURES

Commands:
- pipeline            raw files -> prepared data -> features -> checked outputs
                      (task workbooks, paretic averages, B&B and Motore CSVs),
                      twice: OUTPUT with speeds in m/s, OUTPUT_SPEED_LENGTH
                      with speeds divided by their lever length
- xsens-features      features from data already prepared
  tasks 1-4: 15 features per repetition (five markers)
  B&B, mot1, mot2: 15 features per whole recording (no marker check)
- xsens-repetitions   check the repetition windows only

Elbow: rom_elbow_bending, from quaternions (elbow_rom.py), unsigned.
Shoulder ROM and trunk ranges use Euler angles.

More: ../README.txt (what is calculated), ../CLI_GUIDE.txt (commands),
../CODE_MAP.txt (where in the code).

Rules:
- no production file imports explorations/ or report renderers
- outputs = feature tables, short method and source metadata,
  and a reason for every missing value
- existing workbooks are never overwritten
Local only: narrative reports, old Euler elbow methods.
