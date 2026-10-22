XSENS_CODE - MOVEMENT FEATURES

Commands:
- pipeline            raw files -> prepared data -> 15 features -> checked Excel
- xsens-features      15 features from data already prepared
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
