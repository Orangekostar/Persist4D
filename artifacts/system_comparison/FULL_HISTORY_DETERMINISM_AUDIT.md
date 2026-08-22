# Full-History Determinism Audit

- Status: PASS
- Method: ReScene4D Full-History (Frozen T2 Checkpoint)
- Prefixes: 3 canonical T5 prefixes from distinct reference clusters
- Repeats: 3 per prefix
- Compared: mask/class/query-ID/score fingerprints
- Source commit: `575acc12fbd63f38fc3c16578914b25c2fed8584`

| Reference scene | Master | Fingerprint |
|---|---|---|
| `10b17940-3938-2467-8a7a-958300ba83d3` | `scene0069_00-scene0069_02-scene0069_04-scene0069_03-scene0069_01` | `d74169829bc63f8977e818d3a1169970ca6efc678866c63cddf1106b63b08cd5` |
| `137a8158-1db5-2cc0-8003-31c12610471e` | `scene0079_00-scene0079_09-scene0079_05-scene0079_04-scene0079_03` | `91888582aea6be3f025ab5b26d6eca7c1ce101711f5ee060df8db618bd3dabb4` |
| `280d8ebb-6cc6-2788-9153-98959a2da801` | `scene0119_00-scene0119_02-scene0119_03-scene0119_01-scene0119_04` | `74794be7825450ba5e815fe81bd875fdf463f1dd821ebce73e6825cea5ae5910` |

The identical T2 prefix also passed the complete local/full-history observation fingerprint regression.
