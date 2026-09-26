# 交付与复现命令

先使用交接给出的真实分支/提交，保留自己的数据与基础权重。`assets.local.json` 是本机资产绑定，不入 Git；其每项身份由 `INPUT_MANIFEST.json` 校验。以下命令使用本次已存在的本机路径。

```bash
cd /home/ww/paper5/.worktrees/persist4d-perception-gain-v2
export PERSIST4D_GAIN_V2_ROOT=/home/ww/persist4d_runs/perception_gain_v2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
conda run -n persist4d python -m scripts.perception_gain_v2 status --external-root "$PERSIST4D_GAIN_V2_ROOT"
conda run -n persist4d python -m scripts.perception_gain_v2 report --external-root "$PERSIST4D_GAIN_V2_ROOT"
```

读取已准备的 C0-L 完整替换 bundle，并以已绑定 R1 校验恢复后的全部张量；本命令只验证张量身份，不另称执行真实 panel：

```bash
conda run -n persist4d python -c 'import json, os; from pathlib import Path; from scripts.perception_gain_v2_bundle import load_payload; root = Path(os.environ["PERSIST4D_GAIN_V2_ROOT"]); assets = json.loads((root / "assets.local.json").read_text()); payload, state = load_payload(root / "publication/bundles/C0-L-s45.pt", base_checkpoint=Path(assets["r1_checkpoint"])); print(payload["restored_state_sha256"])'
```

真实直接/重载 panel 的输出 hash 证据在 `resources/PROFILE_SUMMARY.json`。模型分片重组入口为 `python -m scripts.perception_gain_v2_bundle reassemble --bundle MANIFEST --output OUTPUT`，其中 MANIFEST/OUTPUT 应取最终 `RELEASE_PLAN.json` 的确切资产名，不能猜测不存在的下载 URL。

外部发布使用已有 Git/GitHub 授权。若最终 receipt 是 CODE_ONLY，保留本地完整模型和预测包，现有授权可用后运行：

```bash
conda run -n persist4d python -m scripts.perception_gain_v2 publish --external-root "$PERSIST4D_GAIN_V2_ROOT"
```

该命令会创建新的不可变发布快照并选择未占用 tag 后缀，不移动原 tag，也不要求把凭据粘贴到聊天。最终可用 URL 与缺失项只读 external `publication/PUBLICATION_RECEIPT.json`；Git 内 READY 快照不预称 Release 已公开。

训练已到协议终点，不需继续优化；各 `run_summary.json`、`checkpoint_manifest.json` 和 `tables/TRAINING_SOURCE_COUNTS.csv` 给出实际保存步数、draw cursor 与经 plan SHA 复核的已消费来源计数。所有已评价 checkpoint 的分数保存在 `tables/CHECKPOINTS.csv` 与对应 CAL/SEL JSON，未仅保存冠军。原始 checkpoint 与 optimizer 留在 external `training/`，不会通过 Git 分片变相发布基础模型。

独立恢复的实际脚本 SHA、开始/结束时间、退出码、费用和输出路径见 `confirmation/*RECOVERY_AUDIT.json`、`confirmation/LOCAL_EXPORT_MATCHED_RUNTIME_AUDIT.json` 与 `budget/LEDGER.jsonl`。缓存失效不清空费用，不把历史异常覆盖成 PASS。
