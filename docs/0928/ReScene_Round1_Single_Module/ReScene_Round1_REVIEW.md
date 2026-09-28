# 首轮执行指令：交付前审核记录

日期：2026-09-28。审核对象为本包主Markdown与结构化设计YAML，不是服务器代码或新训练结果。

## 审核依据与范围

直接读取了上一份24篇论文路线全文，并再次通过GitHub读取固定提交的原生soft输出、mask物化、官方metric适配、LOCAL入口与V2结果解释；确认V2远端分支仍指向1ddab2a。重新查阅Mask Scoring R-CNN、SoftGroup、SyncVIS原文，以及作者stmetrics的pairwise/enforcement与按分数匹配实现。

本次没有重新阅读全部24个作者仓库，没有运行新GPU实验，没有访问服务器本地checkpoint，没有开发或推送用户仓库。主文件中的数值超参是前瞻试验设计，不是从论文搬来的已验证最优值。

## 已修正的关键歧义

| 项目 | 最终处理 |
|---|---|
| 第一轮与组合边界 | 删除首轮2×2、组合训练、最终组合部署；只留下独立模块元数据 |
| 六模块误认为都能叠加 | 明确Q是评分插槽替代方案，M是mask插槽替代方案；M1是监督变化 |
| M2看起来像先叠M1 | M2从R1新初始化，共用监督是实验控制，不加载或调用M1头 |
| M1同时改目标分配和空目标 | 首轮不做多对一扩展；M0/M1共享一对一分配，仅改变未匹配监督 |
| 训练梯度断开 | 不使用detach/二值化materialize作为训练loss路径；仅用于官方输出评价 |
| 评分与分割收益混在一起 | Q锁mask/class/candidate；M锁score/class/candidate；不二次NMS/过滤 |
| 评分的0步等价性 | Q0是未训练评分，不声称等于B0，不作为已学增益；disabled才是原分数 |
| 同一raw query多class | 用query/class/retained-index构成candidate键，M头共同使用类别条件 |
| 质量标签循环依赖分数 | 用官方pairwise重叠和有效性，不从score-sorted TP/FP取监督 |
| 类别概率重复相乘 | 质量标签包含类别相容性，输出不再乘父score或类别概率 |
| 单扫描与双扫描切片混淆 | T1独立文件读取及前向，unique physical scan去重 |
| M1正例权重暗中放大 | 三M臂共享candidate平均分母，M1取消的负mask项置0，不改分母为正例数 |
| 只选一个冠军 | 六臂各自CAL选点、各自SEL验证；正收益可多项保留 |
| 开发正结果当正式胜出 | 本轮证据上限DEVELOPMENT_REPLICATED，不调用PB/LOCAL正式确认 |
| 发布授权再卡小模型 | 实测≤20MiB小头连同模式文件直接入Git；必要大包缺权限如实CODE_ONLY |

## 静态与数值一致性检查

以下仅检查执行设计内部是否一致，以及简单公式性质；它们不能证明实现正确、实验会涨点或服务器资产可用。

- `fixed_parent_commit`：通过。
- `six_explicit_arms`：通过。
- `single_runtime_module`：通过。
- `all_arm_definitions_present`：通过。
- `frozen_parent`：通过。
- `two_mutually_exclusive_slots`：通过。
- `M0_M1_common_geometry_pooling`：通过。
- `M1_M2_same_target_policy`：通过。
- `no_multi_to_one_confound`：通过。
- `scoring_not_multiplied_twice`：通过。
- `no_score_sorted_quality_labels`：通过。
- `actual_T1_forward`：通过。
- `fixed_steps_and_checkpoints`：通过。
- `all_six_SEL`：通过。
- `multi_candidate_shortlist`：通过。
- `no_formal_or_combo_run`：通过。
- `budget_sum`：通过。
- `head_batch_size`：通过。
- `fresh_replication_identity`：通过。
- `modeled_gradient_boundary`：通过。
- `threshold_score_monotonicity_example`：通过。
- `zero_geometry_residual_formula`：通过。
- `AP_percentage_point_scale`：通过。

## 执行中必须实际确认、不能由本次审核替代的前提

实际运行前仍需定位R1/Concerto、核对已安装stmetrics版本、数据/ignore映射与CAL/SEL完整分母；新CLI和新头尚待Codex实现。head梯度、零残差full-res等价、原native输出一致、固定点复核和Git/Release可访问性，必须由真实工具执行证明。

本文件不将文档检查数当作研究成果。若首轮6臂无正收益，应保留负结果，不擅自扩大模块数量或开始组合。
