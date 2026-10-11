# Persist4D 实验交接

实验提交A：`8146b95de8b0973814bf1adb213f65a86f0ee286`。交接提交B包含本页与最终文件清单，B的自身SHA只记入仓库外发布回执。
分支：`research/persist4d-cvpr-evidence-20261011`；tag：`persist4d-cvpr-evidence-20261011-v1`。
本页记录已完成的测量和资产核验；实际push及远端完整SHA由最后一步外部回执验证。

## 已完成的范围

E0/E1/E2/E3均完成29,700次更新，种子45；40 CAL、6 SEL与3560个正式输入均完整。
机制152项、8项新进程重复、12项部署测量及两个检查点各129条记忆对照全部完成。
所有GPU队列已退出并清空ACTIVE。无需恢复或重新启动训练。

[主报告](REPORT.md)、[逐项验收](REQUIREMENT_REVIEW.md)、[科学审查](../../ccfa-review-reports/2026-10-11-cvpr-readiness.md)
及[中文实验补充稿](../../manuscript_updates/native_long/main.pdf)保留全部正负结果和归因边界。
科学状态为`COMPLETE_WITH_PROVISIONAL_GAIN`，投稿判断仍为`NOT_READY_FOR_CVPR`。
E1相对E0的正式PB S_long为+1.8417 pp；E3相对同一步E1为+0.4847 pp，相对最佳E1为−1.6915 pp。
两个检查点下局部持久记忆改善gap恢复召回和局部任务AP，但均未追平完整前缀AP。
只有一个训练种子；E3锁定点尚处T2课程阶段；固定新进程存在张量非重复。

最终成本1713.8830547291614 GPUh，包含失败尝试；上限1800，余额86.11694527083863，
其中保留8 GPUh。完整第二种子配对预测1034.5019 GPUh，因此未执行。
[成本审计](evidence/COST_AUDIT.json)与191项新增事件账本和历史基数共同构成完整账目。

## 下载与核验

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone --branch persist4d-cvpr-evidence-20261011-v1 \
  git@github.com:Orangekostar/Persist4D.git Persist4D-evidence
cd Persist4D-evidence
git lfs pull
```

五份完整非编码器任务权重及必要buffer共535,331,523字节，已经实际上传LFS并从独立空缓存下载核验。
资产为E0/5940、E1/23760、E1/2970、E2/29700、E3/2970；E2负结果权重同样保留。
加载器、环境与独立编码器SHA见[部署说明](deployment/README.md)。
原始扫描、逐点GT、预训练编码器不重分发，完整optimizer checkpoint仍保留在运行存储，
部署资产不能当作optimizer恢复文件。

`evidence/official_metric_states.zip`包含1093份官方充分状态：CAL/SEL 230、正式596、记忆184、机制75、重复8。
`evidence/measurement_records.zip`保留444份轻量原始测量回执；训练日志包保留109份文件及所有尝试。
官方指标档案76,909,444字节已实际上传LFS，并从独立空缓存下载核验。
每个档案均有成员及整体bytes/SHA清单；`ARTIFACT_MANIFEST.json`覆盖本次交付路径并排除自身hash。
无需点云或模型即可复算已保存的官方指标，实际示例见[指标说明](evidence/README.md)。
重新推理则必须使用原许可数据、锁定权重、数值源和运行协议，不能承诺逐张量重现。

## 后续科研工作

先定位固定新进程的数值非重复，再在共同修复版本上重新验证；不得放宽hash或替换原始结果。
等参数无条件控制、完整独立种子、更多外部场景、长历史/容量饱和与同条件外部强基线尚缺。
原始双复访旧权重结果不混入本轮表格，GT辅助ReScan探针不称官方端到端复现。
当前中文稿是实验整合材料，尚非完整匿名CVPR投稿包。

## 历史与发布身份

初始192 GPUh预检报告保留在旧tag `rescene-native-long-retrain-v1-r2`，对应提交
`a3830a1f1efe66a55df8b147d86941b85165c4cc`。其中零步训练及CODE_ONLY属于历史状态。
新预算锁SHA为`ba1cdf28e8c057315f7a883d43490e2a41e1d1f94f7d566d515f03f72f5d9c5d`。
本次不改写旧tag或旧发布回执。

发布回执位置：运行根的`publication/PUBLICATION_RECEIPT.json`，记录A/B、新tag、实际远端SHA与资产核验。
GitHub固定入口：

- [报告](https://github.com/Orangekostar/Persist4D/blob/persist4d-cvpr-evidence-20261011-v1/artifacts/native_long_postselection_v1/REPORT.md)
- [交接](https://github.com/Orangekostar/Persist4D/blob/persist4d-cvpr-evidence-20261011-v1/artifacts/native_long_postselection_v1/HANDOFF.md)

具体运行根只用于原环境追溯，不替代以上可取得的Git/LFS交付。
