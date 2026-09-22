# Persist4D Perception Gain V1 Final Report

Experiment content commit: `9e537358b4f78d48f037c75ea1aa040125cb12e5`.

- EXECUTION: `PARTIAL_WITH_BLOCKERS`
- PB_ALL_T_VS_NATIVE_FH: `N/A`
- PB_ALL_T_VS_D0: `N/A`
- DEFAULT_DEPLOYMENT: `UNCONFIRMED`
- PUBLICATION_PHASE: `READY`

## 表 1. 任务、实际步数与运行状态

| 任务 | 实际 optimizer updates | 状态 | 阻塞 |
| --- | --- | --- | --- |
| bind | N/A | PASS |  |
| foundation | N/A | PASS |  |
| scorer | 500 | PASS |  |
| perception_pilot | {'C0': 750, 'S-BAL': 350} | BLOCKED | NFS data workers blocked long enough for rank 0 to hit the 1800000 ms NCCL ALLREDUCE watchdog timeout |
| perception_full | N/A | NOT_RUN |  |
| perception_select | N/A | NOT_RUN |  |
| refinement | N/A | NOT_RUN |  |
| final_lock | N/A | NOT_RUN |  |
| replication | N/A | NOT_RUN |  |
| confirm | N/A | NOT_RUN |  |
| profile | N/A | NOT_RUN |  |
| report | N/A | NOT_RUN |  |
| publish | N/A | NOT_RUN |  |

## 表 2. CAL/SEL 各臂与 C0 配对对照

| 角色 | 架构 | update | T2 | T3 | T4 | T5 | 对照 | checkpoint SHA256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAL | C0 | 0 | 0.362623 | 0.291813 | 0.246709 | 0.222703 | R1 | 629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd |
| CAL | C0 | 250 | 0.353755 | 0.275942 | 0.235413 | 0.217662 | R1 | 362a8575fb0d011b079b60af4eb92f57fcb5734484230c76c468bb9718007b87 |
| CAL | C0 | 750 | 0.334286 | 0.280689 | 0.225745 | 0.216750 | R1 | 384884e3e86de79b5ca92661aa887b1f34612d2d8c90e93a66bbf70b87e30143 |
| CAL | Q-SEM | 0 | 0.360675 | 0.295631 | 0.240839 | 0.222269 | C0-paired | f2c9122473ea332d55fca73ce6b01d71d58a75c0b96be7fb56e8892ca1c57173 |
| CAL | S-BAL | 250 | 0.348642 | 0.277610 | 0.239545 | 0.222650 | C0-paired | 9b47b174c1048167fc8156ea2a327692dabae947265adc9193f787d6c7c857f6 |

同一架构只报告冻结 checkpoint；未执行项保持 `NOT_RUN`，不改写为实验失败。

## 表 3. LOCAL-T2 同人口结果

| 方法 | T2 t-mAP | T3 | T4 | T5 | 覆盖 |
| --- | --- | --- | --- | --- | --- |
| NOT_RUN | N/A | N/A | N/A | N/A | 0/0 |

历史参考值不与本轮同人口结果合并；历史证据仍在上游 artifact 中。

## 表 4. PB 各 T、固定起点与差值

| 方法 | T2 | T3 | T4 | T5 | 覆盖 | ΔD0-T2 | ΔD0-T3 | ΔD0-T4 | ΔD0-T5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NOT_RUN | N/A | N/A | N/A | N/A | 0/0 | N/A | N/A | N/A | N/A |

严格胜出仅使用完整 PB 覆盖；不完整覆盖时比较值为 `N/A`。

## 表 5. 每 reference、每 seed 与 ADDITIONAL 人口

| 人口 | 方法 | seed | reference | T | t-mAP/配对均值 | 单位数/状态 |
| --- | --- | --- | --- | --- | --- | --- |
| NOT_RUN | N/A | N/A | N/A | N/A | N/A | 0 |

ADDITIONAL 固定终点人口为 T2 111 单位/40 references、T3 77/23、T4 32/8；没有伪造 T5。

## 表 6. 真实成本、预算、发布与阻塞

| 项目 | 实际 | 上限 | 单位 |
| --- | --- | --- | --- |
| foundation_and_evaluation_gpu_hours | 3.653670 | 32 | GPU-hour |
| perception_training_gpu_hours | 8.150158 | 136 | GPU-hour |
| refinement_gpu_hours | 0.000000 | 16 | GPU-hour |
| profiling_gpu_hours | 0.000000 | 4 | GPU-hour |
| new_cache_bytes | 101166032 | 32 | bytes / GiB limit |
| profile_rows | 0 | N/A | rows |
| publication | READY | N/A | status |

本表中的训练/确认/profile 成本来自真实执行状态。单次部署延迟不包含离线训练或缓存生成；profile 明确排除磁盘冷读与官方 metric 计算。
