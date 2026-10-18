# OGR actual execution and reproduction

Actual charged children, physical cards, stdout paths and exit codes: COMMANDS_EXECUTED.json. The evidence directory preserves failed and successful attempts. Commands below are entry points; a listed entry point alone does not assert execution.

```sh
python -m scripts.ogr_campaign resolve --config conf/observation_grounded_v1/default.yaml
python -m scripts.ogr_campaign diagnose --config conf/observation_grounded_v1/default.yaml
python -m scripts.ogr_campaign probe --config conf/observation_grounded_v1/default.yaml
python -m scripts.ogr_campaign lock --config conf/observation_grounded_v1/default.yaml
python -m scripts.ogr_campaign run --config conf/observation_grounded_v1/default.yaml
python -m scripts.ogr_campaign finalize --config conf/observation_grounded_v1/default.yaml
python -m scripts.ogr_delivery publish --config conf/observation_grounded_v1/default.yaml
```

A fixed-retry entry was executed only after its specific regression and preserved repair receipt. No formal endpoint or 44-job completion is implied by the probe records.
