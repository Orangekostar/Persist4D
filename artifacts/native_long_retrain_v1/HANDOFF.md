# 当前交接入口

此文件此前描述初始预检交付。后续用户授权1800 GPUh集群预算后，四条轨迹均已完成29,700更新，
不再是零步训练或仅代码交付。五份任务权重已实际上传LFS并独立下载核验。

当前结果见[实验报告](../native_long_postselection_v1/REPORT.md)及
[逐项验收](../native_long_postselection_v1/REQUIREMENT_REVIEW.md)。
新分支为`research/persist4d-cvpr-evidence-20261011`，新tag为`persist4d-cvpr-evidence-20261011-v1`。
实验A与交接、下载和复现命令见[本轮最终交接](../native_long_postselection_v1/HANDOFF.md)。
原训练控制器和全部评估队列正常结束，已无活动GPU任务。
实际远端提交SHA由仓库外最终发布回执记录；本页不嵌入自身提交SHA。

[历史预检交付](https://github.com/Orangekostar/Persist4D/blob/rescene-native-long-retrain-v1-r2/artifacts/native_long_retrain_v1/HANDOFF.md)
仍由旧tag保存；不改写旧tag，也不沿用其中的零步/旧预算/恢复命令作为当前状态。
