# CrossWindow fixed-U consistency audit

With the historical assignment `U` frozen, the expanded objective is

`J(Z) = sum(q,k) Z(q,k) B(q,k) + lambda sum(q,i,k) Z(q,k) U(i,k) O(q,i)`.

Collecting terms indexed by `(q,k)` gives

`J(Z) = sum(q,k) Z(q,k) [B(q,k) + lambda sum(i) U(i,k) O(q,i)]`.

The feasible set is unchanged: every query is assigned to at most one real
anchor or its private dummy, and every real anchor is used at most once.
Therefore the bounded W=2 contract contains no second independent consensus
optimization. Exhaustive 2-3 group partial-assignment checks are recorded in
`status.json`; E3 is completed as `SKIPPED_EQUIVALENT`.
