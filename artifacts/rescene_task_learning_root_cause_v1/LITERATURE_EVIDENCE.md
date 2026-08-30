# Literature Evidence For Local Diagnostics

Only primary papers and official repositories are used. Repository revisions are frozen observations for this task; concepts motivate diagnostics but do not authorize an architecture port.

| Work | Venue | Primary source | Official code revision | License status | Repository mapping |
| --- | --- | --- | --- | --- | --- |
| LaSSM: Efficient Semantic-Spatial Query Decoding via Local Aggregation and State Space Models for 3D Instance Segmentation | TCSVT 2026 | [arXiv:2602.11007](https://arxiv.org/abs/2602.11007) | [RayYoh/LaSSM](https://github.com/RayYoh/LaSSM) `4833047c7cb5823ee6634c9da6e63b4b91840c54` | MIT | Motivates measuring geometry-only query initialization and then testing the existing `use_np_features=true` switch. |
| CompetitorFormer: Mitigating Query Conflicts for 3D Instance Segmentation via Competitive Strategy | CVPR 2026 | [CVF paper](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_CompetitorFormer_Mitigating_Query_Conflicts_for_3D_Instance_Segmentation_via_Competitive_CVPR_2026_paper.html) | [DuanchuWang/CompetitorFormer](https://github.com/DuanchuWang/CompetitorFormer) `9efc41ec090e08496c31970eddc0215fc4e91b87` | MIT | Motivates a query-conflict diagnostic before any competition module. |
| Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation | CVPR 2025 | [CVF paper](https://openaccess.thecvf.com/content/CVPR2025/html/Lu_Relation3D__Enhancing_Relation_Modeling_for_Point_Cloud_Instance_Segmentation_CVPR_2025_paper.html) | [Howard-coder191/Relation3D](https://github.com/Howard-coder191/Relation3D) `a2fc84bd7a907bb935760c668910fa746940c130` | No license file found at the pinned revision | Motivates superpoint-feature diagnostics and conditional use of the existing `scatter_type=adaptive` implementation. |
| Mask-Attention-Free Transformer for 3D Instance Segmentation | ICCV 2023 | [CVF paper](https://openaccess.thecvf.com/content/ICCV2023/html/Lai_Mask-Attention-Free_Transformer_for_3D_Instance_Segmentation_ICCV_2023_paper.html) | [JIA-Lab-research/Mask-Attention-Free-Transformer](https://github.com/JIA-Lab-research/Mask-Attention-Free-Transformer) `4b5048c0e08c2fc42f660dfea3209043179ace1b` | No license file found at the pinned revision | Motivates measuring early mask-attention recall and all-masked reset events before relaxing attention masks. |

## Scope

`use_np_features` and adaptive scatter are ReScene-native switches, not implementations of LaSSM or Relation3D. Query competition and attention-mask changes remain high-risk and require a committed diagnostic gate plus a separate design. Repositories without a declared license are used only as public conceptual evidence; no source is copied.
