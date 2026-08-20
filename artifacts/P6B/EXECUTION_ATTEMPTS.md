# P6-B Execution Attempts

This supplemental ledger records execution history outside the canonical artifact
manifest. It does not change any metric, gate, or selected configuration.

1. The first final-evaluation process used frozen selection commit `caf47ad`. It
   reached `heldout_evaluation_complete` but failed before artifact publication
   because JSON object key ordering was incorrectly treated as CSV schema order.
   No final metric was emitted, inspected, or used for configuration selection.
   The external log is `external:paper5/logs/p6b_final_20260820.log`.
2. The packaging defect was reproduced by a failing unit test and fixed in
   `04efe5f`. The full tuning sweep was rerun from clean source `976fccd`.
3. The rerun selected the same configuration, `p6b-ab4b4cfae20de568`. Its tuning
   payload is identical to the first sweep except for source binding.
4. The successful held-out artifact was generated from frozen selection commit
   `0036186` and produced the terminal decision `P6B_STOP`. The external log is
   `external:paper5/logs/p6b_final_retry_20260820.log`.
