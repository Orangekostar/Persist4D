# MPR-V1 实际命令与复现入口

在原仓库环境中绑定合法取得的数据、固定父包及编码器。已冻结的选点和结果保持原身份；以下汇总命令只读取对应评测命名空间。

```sh
python -m scripts.revisit_prompt_campaign resolve --auto
python -m scripts.revisit_prompt_campaign plan --dry-run
```

尚未完成的工作使用原任务恢复入口；只恢复身份一致的项目，不重新训练已完成轨迹：

```sh
python -m scripts.continue_revisit_prompt_sparse
```

冻结发布前从实际状态汇总；已发布 A/B 的产物不原地改写：

```sh
python - <<'PY'
from scripts.revisit_prompt_assets import local_assets
from scripts.revisit_prompt_sparse_policy import campaign_policy, load_policy
from scripts.revisit_prompt_campaign import summarize
assets = local_assets()
with campaign_policy(assets, load_policy(assets)) as scoped:
    summarize(scoped)
PY
```

真实发布使用本轮独立引用：

```sh
python -m scripts.publish_revisit_prompt --execute --branch research/persist4d-mpr-v1-formal-20261014 --tag persist4d-mpr-v1-formal-20261014-v1
```

父包实体取得：

```sh
git lfs pull --include=artifacts/native_long_postselection_v1/deployment/E1-23760.pt --exclude=
```

完整的依赖取得、严格加载和官方状态离线复算命令见 [DEPLOYMENT_PROTOCOL.md](DEPLOYMENT_PROTOCOL.md)，固定来源和哈希见 [DEPENDENCY_ACQUISITION.json](DEPENDENCY_ACQUISITION.json)。加载和前向均须保持其中的 V4 推理上下文。
模型文件与实际 optimizer 恢复文件分别见 MODEL_MANIFEST.json；delta 不含 optimizer，最终 resume 的进度为原 N 边界。官方状态离线复算不运行 GPU 推理。
