# Code Bindings

| Requirement | Implementation | Actual evidence |
|---|---|---|
| Encoder-only/task-fresh | models/pointcept.py::_load_state_dict | SOURCE_AND_INITIALIZATION.json |
| Real population/draw/RNG | datasets/native_long_dataset.py | POPULATION.json; DATA_CONTENT_BINDING.json |
| LOW/FULL inverse correctness | datasets/pointcept_utils.py | initialization/REAL_MIXED_BATCH.json; collation regression |
| Fixed-count stage sampling | models/native_long_modules.py; models/rescene.py | diagnostics/SAMPLING.csv |
| Late query-conditioned F | models/native_long_modules.py; ReScene.update_mask_features | REAL_FEEDBACK_IDENTITY.json; FEATURE_UPDATE.csv |
| Boundary resume | trainer/native_long_trainer.py | actual Lightning split-resume fixture, draw/LR/tensor equality |
| raw_sum/12aux/backward | scripts/native_long_runtime.py::training_measurement | resources/COST_MEASUREMENTS.json |
| Budget/no doomed training | native_long_campaign.freeze_cost_plan | RESOURCE_PLAN.json; RUN_STATE.json |
| Checkpoint CAL/SEL | scripts/native_long_execution.py | selection regression only; real trajectories not run |
| Exact task/buffer loader | scripts/native_long_assets.py | initialization/TASK_BUNDLE_RELOAD_AUDIT.json |

Seed46 pair execution, formal-confirmation integration and selected-model profiling remain unexecuted and are not certified by this budget-limited delivery. The current authorized empty-plan CLI is exercised.
