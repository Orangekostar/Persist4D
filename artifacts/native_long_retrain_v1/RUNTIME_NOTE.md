# 当前运行限制

状态：**FRESH_PROCESS_NONREPEATABILITY_OBSERVED**。
早期预检在每个固定A/BA/ABA协议下的两次结果一致，改变顺序时首先定位到
编码器enc0.block0.cpe.0稀疏卷积；Native算法探针仍保留历史敏感性，未启用该替换。
这些单输入检查不能作为整个总体逐张量可重复的保证。

已训练模型更大的30组机制面板发现2组反馈之前就有数值差异，尽管输入和GT哈希一致。
追加8项独立进程重复后，同模式12个两两比较中6个有上游/官方状态差异，两个输入AP/REC仍一致。
本次新进程异常的具体更早算子尚未定位；不能据早期顺序调查断言是相同根因。

完整证据、推理协议与限制见
[运行限制回执](../native_long_postselection_v1/evidence/FRESH_PROCESS_RUNTIME_LIMITATION.json)、
[重复审计](../native_long_postselection_v1/evidence/MECHANISM_REPLAY_AUDIT.json)及
[当前报告](../native_long_postselection_v1/REPORT.md)。
不放宽哈希容差，不替换原始异常配对，不把AP相同写成逐张量一致。新增训练须先解决原生一致性。

全部训练采用两张A40、每卡batch1、累积16，有效batch32。
该设置来自完整大T5的实测：batch2在精确checkpoint/chunking后仍OOM，batch1完整反传通过。
四条实际轨迹均29,700更新；不改变FP32、TF32关闭、体素大小、Q=100或完整输入来掩盖容量问题。

这里的FP32指Lightning的32-true配置和任务参数精度；原Concerto的FlashAttention内部
显式使用FP16 QKV，再转回输入dtype。见新的NUMERICAL_PRECISION_SCOPE.json，不宣称全算子FP32。
该事实不是对本次数值非重复根因的判断。
