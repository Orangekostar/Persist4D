# Protocol-B Exact T2 Bridge Build Report

## Construction

Each record is the exact canonical first-two-scan prefix of one registered
T5 master. Its target contains only those point rows and transition column 0.
No missing pair is replaced by another pair or by reverse order.

## Frozen Sources

- `protocol_b_manifest`: `repo:artifacts/P6A/protocol_b_manifest.json` / `246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe`
- `source_t5_database`: `repo:data/processed/rio/sequence_database_sliding_5.yaml` / `252363f76524bb7eeff9f65b303aadda67dcd2646477daae1ac90f7f53398290`
- `validation_scan_metadata`: `repo:data/processed/rio/validation_database.yaml` / `cc69f9de660e6be5b6aee739ee2580b614f132ad8c3a42ce2e301b4d4dbb1906`
- `official_sliding_t2_database`: `repo:data/processed/rio/sequence_database_sliding_2.yaml` / `974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416`
- `frozen_t2_inventory_audit`: `repo:artifacts/tmap_root_cause_v2/PROTOCOL_SHIFT_AUDIT.md` / `f861aa4f65e618bc34985e3fdb9e08bef79db81131c0d91907c3a43d8d746ce3`

## PB0

- Canonical prefixes: `43/43`
- Existing T2 overlaps: `14/14`
- Exact overlap parity: `14/14`
- Pair substitutions: `0`
- Reverse substitutions: `0`
- Future-stage leakage: `0`
- Validation and supervised: `43/43`

Gate PB0: **PASS**.
