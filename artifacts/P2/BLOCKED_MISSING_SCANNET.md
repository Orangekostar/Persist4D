# P2 ScanNet Prerequisite Blocked

Status: `BLOCKED_MISSING_SCANNET`

Official ReScene4D-C training requires the authorized ScanNet v2 release mixed with 3RScan. The current gate does not permit a 3RScan-only run to be labeled an official reproduction.

## Exact Missing Prerequisite

- Official split coverage must be train=1201, validation=312, test=100.
- Raw asset coverage is 0/1613 scenes; missing assets=7765.
- Processed DB coverage is 0/1613 scenes.
- Processed NPY coverage is 0/1613 scenes.
- NYU40 instance taxonomy status is `fail`.
- Real mixed-dataset instantiation status is `blocked_prerequisites`.
- RIO active T=2 sequences without an NYU40-18 instance: 7 (`scene0242_00-scene0242_01, scene0242_01-scene0242_02, scene0242_02-scene0242_00, scene0245_01-scene0245_02, scene0439_00-scene0439_02, scene0439_01-scene0439_00, scene0439_02-scene0439_01`)
- Blocking error codes: `label_database_invalid_or_missing, label_database_validation_mapping_mismatch, metric_class_ids_mismatch, metric_class_labels_mismatch, metric_dataset_name_mismatch, metric_taxonomy_invalid_or_missing, processed_database_count_mismatch, processed_database_missing, processed_database_scene_missing, processed_global_file_missing, raw_scene_assets_incomplete, rio_active_sequence_supervision_empty, scannet_label_map_missing, scannet_processed_root_missing, scannet_raw_root_missing`.

Required inputs must be obtained under the official ScanNet terms, preprocessed into the repository schema, and then re-audited. Unauthorized mirrors are not an acceptable substitute.

## Not Executed As Official Reproduction

- formal topology benchmark
- official smoke test
- official tiny overfit
- formal 450-epoch training
- formal checkpoint
- G2 metrics

No formal training or metric verdict is recorded while this prerequisite remains blocked.
