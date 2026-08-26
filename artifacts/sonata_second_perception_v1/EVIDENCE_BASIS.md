# Sonata Second-Perception Evidence Basis

Captured on `2026-08-26`; external values below remain external references and
are not local measurements.

## ReScene4D

- Paper: Emily Steiner, Jianhao Zheng, Henry Howard-Jenkins, Chris Xie, and Iro
  Armeni, "ReScene4D: Temporally Consistent Semantic Instance Segmentation of
  Evolving Indoor 3D Scenes," CVPR 2026.
- arXiv: `https://arxiv.org/abs/2601.11508v2`; arXiv metadata updated
  `2026-04-01T23:18:50Z` and identifies CVPR 2026.
- CVPR source: `https://openaccess.thecvf.com/CVPR2026`.
- DOI status: no exact-title DOI was discoverable in Crossref at capture time;
  no DOI is invented or substituted.
- Official code: `https://github.com/GradientSpaces/rescene4d`, revision
  `fb2fe42eb8f1e926567c48eea9acb874e608ee10`.
- README SHA-256:
  `4550760cce90bc372175cc9638148c6cf6d581058b24c590bf0c88c27a31d070`.
- Official task checkpoint status at that revision: `Coming soon`.
- Paper-reported ReScene4D-S: t-mAP 33.2%, standard mAP 40.9%.
- Paper-reported Sonata without temporal sharing: t-mAP 29.7%.

## Sonata

- Paper: "Sonata: Self-Supervised Learning of Reliable Point
  Representations," CVPR 2025; `https://arxiv.org/abs/2503.16429`.
- Official code: `https://github.com/facebookresearch/sonata`, revision
  `18c09ff8d713494f78a8213792262b910977a65d`.
- README SHA-256:
  `345a3dd5f4d2712cb427e768e5159a4b05ac30588b3aaebc5efeffdcdd15667c`.
- Official model repository: `facebook/sonata`, immutable revision
  `df99897472c09f91ba9288da0a034aacffc0b010`.
- Declared `sonata.pth`: 434,008,287 bytes, LFS SHA-256
  `c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50`.
- Code license: Apache-2.0. Weight license: CC-BY-NC-4.0.
- The checkpoint is encoder-only PTv3; the ReScene decoder is not an official
  pretrained Sonata task decoder.

## Local Frozen Evidence

- Workspace branch: `research/persist4d-sonata-second-perception-v1`.
- Start commit: `e5d7f4e96fedc76c0c6d414ab293f54909c61df3`.
- V3 final report SHA-256:
  `265efb46684727194f76eff88b2a662dc2de1369987c80cdf43a6d320f9acf1d`.
- V3 final manifest SHA-256:
  `73f87bd094ef13f5dbf555c71fb7604ca42ea7bcef1d6dd3f31fc9e1eae99c42`.
- Current Concerto task checkpoint SHA-256:
  `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`.
- Current Concerto pretrained weight: 433,987,358 bytes, SHA-256
  `845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`,
  immutable source revision `c31f993a56129f2ba9c5d06a35957e3f05bff710`.

## Discoverable Third-Party Revisions

The frozen P2 environment manifest records:

| Component | Revision | License/status |
|---|---|---|
| ReScene4D | `fb2fe42eb8f1e926567c48eea9acb874e608ee10` | upstream code |
| Sonata | `18c09ff8d713494f78a8213792262b910977a65d` | Apache-2.0 |
| Concerto | `10a7d17cff4dddff028f1522c2e72de4c4515df7` | Apache-2.0 |
| ScanNet tools | `3830fce7f8b2e48ef047ef7fd76ea5f62903f51c` | source revision |
| detectron2 | `b4a4a3bd136852dae5fb1de37978dee412653e31` | Apache-2.0 |
| stmetrics | `640e34c2dd15c8e1a5061f4e66aa4fb6a5da9a5f` | MIT |

The SS1 and SS2 manifests must independently bind the actual weight, runtime,
and data used in this branch; this document does not authorize training.
