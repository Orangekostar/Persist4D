# Native Runtime Note

Status: RUNTIME_CONDITIONAL; repeats within each fixed A/BA/ABA protocol are identical. Input, inverse/segment mappings and embedding are identical across orders. The first localized difference is encoder enc0.block0.cpe.0 sparse convolution. A Native sparse-algorithm probe also retained history sensitivity; that algorithm is not enabled.

Use a fresh process and sorted (horizon,input_id) evaluation order, per-input seed hash(9001,input_id), FP32, TF32 off, highest matmul precision. No 94-input replay. Training keeps frozen encoder modules in train mode.

The initialized TRAIN A probe yields t-AP=0 in A, BA and ABA, but BA changes candidate lineage. This single-input diagnostic does not resolve trained-model accuracy or confirm any gain. Official recall can be undefined at initialization; no NaN was replaced by zero.

Largest complete T5: batch2 OOM both before and with exact non-reentrant F chunk checkpointing; batch1 complete backward/update passed. All arms use batch1/rank, world2, accumulation16. The fixed OneCycle horizon and effective batch32 remain unchanged.

Resource deviation: two early CPU resume fixtures called a legacy all-device RNG helper and initialized three visible GPU contexts. Conservatively charged 3 cards times their full 5.52s/6.28s process durations. The new hook captures only its rank-local GPU, and final CPU checks hide CUDA.
