# OGR-V1 implementation design

Authority: artifacts/observation_grounded_v1/EXECUTION_INSTRUCTION.md, exact supplied 2026-10-18 instruction, BASE 758b5aa4bd2e9119ae5e05f725faa360c2120d88. The user authorizes implementation, locked experiments and actual Git/LFS publication without further questions. No previous MPR result, lock, source or ledger is edited.

## Model and state

ObservationGroundedReScene subclasses RevisitPromptReScene with a fresh M4 front end and separate ogr_variant. Current dense features are captured before prompt injection. Differentiable FP32 softmask pooling uses current full-resolution point-count weights and detached backbone features. IdentityEvidenceHead uses identity-initialized Wc/Wh plus null; foreground class priors are detached. R-QFB/R-OBS share regional P, historical mixture, parameter initialization and feedback placement; only the direct value differs. One zero-initialized 384→128→128 feedback MLP runs after the first decoder cycle, only on source prompt queries. The original forward stays unchanged.

ObservationEvidenceState is detached prediction-only K100 state. Final valid current support is routed and committed once before common/geometry/observation state updates. Partial assignment uses per-row dummy columns and complete EntityRoute commitments. No identity probability changes official scores or confidence. M0 and old M4 use their original read semantics in the same wrapper; FH-COMMIT0 changes the causal input window only.

## Supervision and training

A criterion wrapper supplies constrained per-layer assignment while reusing original task and contrastive loss semantics. Canonical reliable source is forced once; duplicate source queries are excluded from CE numerator/denominator and mask matching. Unknown sources remain ordinary matching candidates. Group-positive identity supervision excludes unknown history columns from supervised normalization; null requires every effective history identity to be known. No GT enters model state or retrieval.

All frozen parent parameters/buffers remain eval. The new builder explicitly pins Native convolutions. The V4-derived context restores patches and uses ordered sparse operations exclusively inside the inherited frozen-backbone no_grad; mask pooling, identity and feedback retain autograd. New prefixes are model.mpr_prompt., model.ogr_identity., model.ogr_feedback. Expected counts 84224/116993/182785/182785 are checked against actual active tensors.

## Runtime and evidence

An independent resolver derives paths and prior from existing checked assets/ledgers. One shared flock journal reserves unique job costs atomically across at most two single-GPU workers. Prior excludes this run; spent/reservations are deducted once; reserve8, lifetime1823, run50 and each specified phase cap remain enforced. Physical GPUs are rechecked before each launch, and no other process is stopped.

Diagnose executes the fixed 34 CAL long inputs with M0 shared history; intervention counts and NA are explicit. Four real 20-update probes determine train costs and TRAIN-only lambda. The cost forecast includes all 44 evaluation jobs before N is locked. N=1200/800/400 is cost-selected before CAL scores, otherwise the exact NO_FEASIBLE_COMMON_N branch is delivered. Second seed247 is mandatory within the locked plan even for negative first-seed results. Full-prefix and CONFIRM have explicit independent data contracts; unknown exposure never becomes independent confirmation.

## Completion

Four focused test files cover data/causality, learning/gradient, runtime/route, and assets/publication. Bounded real GPU checks, probes and failures are charged. All scheduled points, failures, counts, sufficient metric states and negative results are preserved. Physical-reference paired bootstrap uses original AP and original NA rule. Three actual-data figure families only. One primary twelve-item content review precedes A/B commits and an annotated tag. Final remote verification downloads only the new final assets once into an empty LFS cache; the external receipt truthfully separates Git/LFS and unavailable Release API.
