# Ordered sparse convolution diagnostic

The original C3 training sources, 24 completed checkpoints, v1/v2 locks and all
failed evaluation repeats remain immutable. The first stable-centroid full
control failed its original 0.2 pp long-repeat guard. Later equal captures do
not replace that failure or establish a root cause.

Primary agent owns the numerical design, implementation and validation. Luna
only monitors explicitly mapped files and processes.

1. Preserve the twelve deferred captures and their original P0 charges. They
   contain 1085 equal encoder observations per process and no observed tensor
   mutations; their single-input scope does not establish full CAL stability.
2. Test an evaluation-only reference for the observed FP32 Native SubMConv
   operation. Retain KRSC weights, original neighbor pairs, the center product
   and increasing kernel-offset order. Use gather, FP32 torch.mm, addition and
   index_copy with unique destinations. Reject unsupported operations, duplicate
   destinations and gradient-enabled use. Restore the original operation on exit.
3. Compare known sparse neighborhoods and the installed spconv CPU operation.
   Run bounded charged GPU controls before adopting any new formal policy. The
   original Native+stable means+stable centroids remains the comparison base.
4. At most three single-input controls (0.03 GPUh each), then conditionally
   three complete M1-u200 and three complete M3-u300 controls (0.4 GPUh each)
   use the original P0 ledger. Stop on errors or failed controls. Preserve all
   repeats and report correctness, runtime and metric spread independently.
5. A later formal policy requires its own source/runtime lock, separate outputs,
   fresh three-process M0 admission and unchanged selection/replication guards.
   This diagnostic alone does not authorize reusing old metrics under a new name.

This is a mathematical reference and a controlled intervention, not a claim
that the previous intermittent outlier has been causally explained.
