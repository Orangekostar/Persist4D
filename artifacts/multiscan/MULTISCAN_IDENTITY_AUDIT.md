# MultiScan Identity Audit

Status: `NOT_VERIFIED_RELEASE_ACCESS_BLOCKED`

Official source code at `697bc9ec86fb7d34d47cb4cdbddcfc3c7f18c605` writes `inst2obj_id` into generated
instance-segmentation PTH payloads. Official annotation documentation defines
`objectId` as an object's per-scan list index plus one. Those two source facts do
not by themselves prove that a repeated numeric ID denotes the same physical
object across scans.

The real released PTH and annotations could not be opened because both official
release paths require an authenticated, license-accepted session. Therefore no
manual cross-scan example is reported and stable identity is not marked verified.
Synthetic tests enforce local-instance remapping and fail on ID/label conflicts,
but synthetic evidence is not substituted for release evidence.
