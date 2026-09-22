#!/usr/bin/env python3
"""Generate the Perception Gain final report, handoff, and release manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/perception_gain_v1"
DEFAULT_EXTERNAL_ROOT = Path("/home/ww/persist4d_runs/perception_gain_v1")
STAGES = (
    "bind",
    "foundation",
    "scorer",
    "perception_pilot",
    "perception_full",
    "perception_select",
    "refinement",
    "final_lock",
    "replication",
    "confirm",
    "profile",
    "report",
    "publish",
)
HORIZONS = (2, 3, 4, 5)


class ReportingError(RuntimeError):
    """Raised when report inputs or generated delivery files are invalid."""


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.is_file() and not required:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReportingError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise ReportingError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fmt(value: object, *, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value).upper()
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if any(len(row) != len(headers) for row in rows):
        raise ReportingError("report table width differs")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _candidate_rows(artifact_root: Path) -> list[list[object]]:
    rows = []
    paths = sorted(
        set(artifact_root.glob("training/**/evaluation/cal/**/*.json"))
        | set(artifact_root.glob("selection/evaluation/sel/**/*.json"))
    )
    for path in paths:
        summary = _read_json(path, required=False)
        candidate = summary.get("candidate") if isinstance(summary, Mapping) else None
        if not isinstance(candidate, Mapping):
            continue
        metrics = candidate.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        role = summary.get("data_role")
        variant = summary.get("variant", candidate.get("method_id"))
        update = candidate.get("optimizer_update", summary.get("optimizer_update"))
        baseline = (
            "C0-paired" if variant not in {"C0", None} and role == "CAL" else "R1"
        )
        rows.append(
            [
                role,
                variant,
                update,
                *[
                    metrics.get(str(horizon), metrics.get(horizon))
                    for horizon in HORIZONS
                ],
                baseline,
                candidate.get("checkpoint_sha256"),
            ]
        )
    return rows or [["NOT_RUN", "N/A", 0, None, None, None, None, "N/A", "N/A"]]


def _confirmation(artifact_root: Path) -> dict[str, Any] | None:
    return _read_json(
        artifact_root / "confirmation/CONFIRMATION_SUMMARY.json", required=False
    )


def _confirmation_rows(
    confirmation: Mapping[str, object] | None, group: str
) -> list[list[object]]:
    methods = confirmation.get("methods") if isinstance(confirmation, Mapping) else None
    group_methods = methods.get(group) if isinstance(methods, Mapping) else None
    if not isinstance(group_methods, Mapping):
        return [["NOT_RUN", None, None, None, None, "0/0"]]
    rows = []
    for method, record in group_methods.items():
        if not isinstance(record, Mapping):
            continue
        metrics = record.get("metrics")
        coverage = record.get("coverage")
        if not isinstance(metrics, Mapping) or not isinstance(coverage, Mapping):
            continue
        rows.append(
            [
                method,
                *[metrics.get(str(horizon)) for horizon in HORIZONS],
                f"{coverage.get('completed_prefixes', 0)}/{coverage.get('expected_prefixes', 0)}",
            ]
        )
    return rows


def _reference_seed_rows(
    confirmation: Mapping[str, object] | None,
    replication: Mapping[str, object] | None,
) -> list[list[object]]:
    rows: list[list[object]] = []
    methods = confirmation.get("methods") if isinstance(confirmation, Mapping) else None
    if isinstance(methods, Mapping):
        for population in ("PB", "ADDITIONAL"):
            group = methods.get(population)
            if not isinstance(group, Mapping):
                continue
            for method, record in group.items():
                metric_rows = (
                    record.get("metric_rows") if isinstance(record, Mapping) else None
                )
                if not isinstance(metric_rows, Sequence) or isinstance(
                    metric_rows, (str, bytes)
                ):
                    continue
                for metric in metric_rows:
                    if (
                        not isinstance(metric, Mapping)
                        or metric.get("reference") == "all"
                    ):
                        continue
                    rows.append(
                        [
                            population,
                            method,
                            45,
                            metric.get("reference"),
                            metric.get("T"),
                            metric.get("t_mAP"),
                            metric.get("logical_unit_count"),
                        ]
                    )
    if isinstance(replication, Mapping):
        for item in replication.get("comparisons", ()):
            if isinstance(item, Mapping):
                rows.append(
                    [
                        "SEL",
                        item.get("method_id"),
                        46,
                        "all",
                        "T2-T5",
                        item.get("S_mean"),
                        item.get("status"),
                    ]
                )
    return rows or [["NOT_RUN", "N/A", "N/A", "N/A", "N/A", None, 0]]


def _execution_status(state: Mapping[str, object]) -> str:
    completed = state.get("completed_stages")
    failures = state.get("failures")
    if isinstance(completed, Sequence) and "profile" in completed:
        return "COMPLETE"
    if isinstance(failures, Sequence) and failures:
        return "PARTIAL_WITH_BLOCKERS"
    return "IN_PROGRESS"


def _render_report(
    *,
    state: Mapping[str, object],
    run_config: Mapping[str, object],
    confirmation: Mapping[str, object] | None,
    replication: Mapping[str, object] | None,
    profile: Mapping[str, object] | None,
    artifact_root: Path,
    experiment_commit: str,
    publication_phase: str,
) -> str:
    completed = set(state.get("completed_stages", ()))
    failures = state.get("failures")
    failure_by_stage = {
        str(item.get("stage")): item
        for item in failures or ()
        if isinstance(item, Mapping)
    }
    stage_rows = []
    for stage in STAGES:
        payload = state.get(stage)
        updates = None
        if isinstance(payload, Mapping):
            updates = (
                payload.get("pilot_endpoint")
                or payload.get("completed_updates")
                or payload.get("training_updates")
            )
        failure = failure_by_stage.get(stage)
        if updates is None and isinstance(failure, Mapping):
            updates = failure.get("completed_arm_updates") or failure.get(
                "completed_optimizer_updates"
            )
        status = (
            "PASS"
            if stage in completed
            else "BLOCKED" if stage in failure_by_stage else "NOT_RUN"
        )
        stage_rows.append(
            [
                stage,
                updates,
                status,
                failure.get("reason", "") if isinstance(failure, Mapping) else "",
            ]
        )

    local_rows = _confirmation_rows(confirmation, "LOCAL-T2")
    pb_rows = _confirmation_rows(confirmation, "PB")
    pb_methods = (
        confirmation.get("methods", {}).get("PB", {})
        if isinstance(confirmation, Mapping)
        else {}
    )
    d0_metrics = (
        pb_methods.get("D0-R1", {}).get("metrics", {})
        if isinstance(pb_methods, Mapping)
        else {}
    )
    pb_with_deltas = []
    for row in pb_rows:
        method = row[0]
        record = pb_methods.get(method, {}) if isinstance(pb_methods, Mapping) else {}
        metrics = record.get("metrics", {}) if isinstance(record, Mapping) else {}
        pb_with_deltas.append(
            [
                *row,
                *[
                    (
                        float(metrics[str(horizon)]) - float(d0_metrics[str(horizon)])
                        if str(horizon) in metrics and str(horizon) in d0_metrics
                        else None
                    )
                    for horizon in HORIZONS
                ],
            ]
        )
    budget = (
        state.get("budget_used")
        if isinstance(state.get("budget_used"), Mapping)
        else {}
    )
    resolved = run_config.get("resolved_config")
    limits = resolved.get("budget") if isinstance(resolved, Mapping) else {}
    cost_rows = []
    for key in (
        "foundation_and_evaluation_gpu_hours",
        "perception_training_gpu_hours",
        "refinement_gpu_hours",
        "profiling_gpu_hours",
    ):
        cost_rows.append([key, budget.get(key, 0.0), limits.get(key), "GPU-hour"])
    cost_rows.extend(
        [
            [
                "new_cache_bytes",
                budget.get("new_cache_bytes", 0),
                limits.get("new_cache_gib"),
                "bytes / GiB limit",
            ],
            [
                "profile_rows",
                profile.get("measurement_rows") if isinstance(profile, Mapping) else 0,
                None,
                "rows",
            ],
            ["publication", publication_phase, None, "status"],
        ]
    )
    comparisons = (
        confirmation.get("comparisons", {}) if isinstance(confirmation, Mapping) else {}
    )
    deployment = (
        confirmation.get("deployment", {}) if isinstance(confirmation, Mapping) else {}
    )
    verdict = _execution_status(state)
    return f"""# Persist4D Perception Gain V1 Final Report

Experiment content commit: `{experiment_commit}`.

- EXECUTION: `{verdict}`
- PB_ALL_T_VS_NATIVE_FH: `{_fmt(comparisons.get('pb_all_t_vs_native_fh'))}`
- PB_ALL_T_VS_D0: `{_fmt(comparisons.get('pb_all_t_vs_d0'))}`
- DEFAULT_DEPLOYMENT: `{_fmt(deployment.get('recommended_method', 'UNCONFIRMED'))}`
- PUBLICATION_PHASE: `{publication_phase}`

## 表 1. 任务、实际步数与运行状态

{_table(('任务', '实际 optimizer updates', '状态', '阻塞'), stage_rows)}

## 表 2. CAL/SEL 各臂与 C0 配对对照

{_table(('角色', '架构', 'update', 'T2', 'T3', 'T4', 'T5', '对照', 'checkpoint SHA256'), _candidate_rows(artifact_root))}

同一架构只报告冻结 checkpoint；未执行项保持 `NOT_RUN`，不改写为实验失败。

## 表 3. LOCAL-T2 同人口结果

{_table(('方法', 'T2 t-mAP', 'T3', 'T4', 'T5', '覆盖'), local_rows)}

历史参考值不与本轮同人口结果合并；历史证据仍在上游 artifact 中。

## 表 4. PB 各 T、固定起点与差值

{_table(('方法', 'T2', 'T3', 'T4', 'T5', '覆盖', 'ΔD0-T2', 'ΔD0-T3', 'ΔD0-T4', 'ΔD0-T5'), pb_with_deltas)}

严格胜出仅使用完整 PB 覆盖；不完整覆盖时比较值为 `N/A`。

## 表 5. 每 reference、每 seed 与 ADDITIONAL 人口

{_table(('人口', '方法', 'seed', 'reference', 'T', 't-mAP/配对均值', '单位数/状态'), _reference_seed_rows(confirmation, replication))}

ADDITIONAL 固定终点人口为 T2 111 单位/40 references、T3 77/23、T4 32/8；没有伪造 T5。

## 表 6. 真实成本、预算、发布与阻塞

{_table(('项目', '实际', '上限', '单位'), cost_rows)}

本表中的训练/确认/profile 成本来自真实执行状态。单次部署延迟不包含离线训练或缓存生成；profile 明确排除磁盘冷读与官方 metric 计算。
"""


def _portable_commands(artifact_root: Path) -> list[str]:
    path = artifact_root / "EXECUTION_LOG.jsonl"
    if not path.is_file():
        return []
    commands = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        command = row.get("command") if isinstance(row, Mapping) else None
        if isinstance(command, str) and command and command not in commands:
            commands.append(command)
    return commands


def _render_handoff(
    *,
    state: Mapping[str, object],
    run_config: Mapping[str, object],
    roles: Mapping[str, object],
    confirmation: Mapping[str, object] | None,
    profile: Mapping[str, object] | None,
    artifact_root: Path,
    experiment_commit: str,
    publication_phase: str,
) -> str:
    resolved = run_config.get("resolved_config")
    identity = resolved.get("identity") if isinstance(resolved, Mapping) else {}
    lock = _read_json(artifact_root / "selection/FINAL_LOCK.json", required=False)
    recipe = lock.get("recipe") if isinstance(lock, Mapping) else None
    failures = state.get("failures")
    failure_lines = [
        f"- `{item.get('stage', 'unknown')}`: {_fmt(item.get('reason'))}"
        for item in failures or ()
        if isinstance(item, Mapping)
    ] or ["- 无已记录阻塞。"]
    commands = _portable_commands(artifact_root)
    command_lines = [f"- `{value}`" for value in commands] or ["- 尚无已执行命令。"]
    role_values = roles.get("roles") if isinstance(roles.get("roles"), Mapping) else {}
    confirmation_status = (
        confirmation.get("execution_status")
        if isinstance(confirmation, Mapping)
        else "NOT_RUN"
    )
    profile_status = (
        profile.get("status") if isinstance(profile, Mapping) else "NOT_RUN"
    )
    completed = ", ".join(str(value) for value in state.get("completed_stages", ()))
    return f"""# Persist4D Perception Gain V1 Handoff

## 1. 仓库、起点 SHA、工作分支、实验代码/结果提交 SHA

仓库 `Orangekostar/Persist4D`；起点 `{_fmt(identity.get('parent_commit'))}`；分支 `{_fmt(identity.get('branch'))}`；实验内容提交 `{experiment_commit}`。

## 2. 本轮目标与最终结论：哪项是实测、哪项仍未确认

执行状态 `{_execution_status(state)}`。已完成阶段：{completed}。确认状态 `{confirmation_status}`；未完成项不作胜出结论。

## 3. 已改文件与函数、训练/推理数据流

唯一代码映射见 `CODE_BINDINGS.md`。训练为 ScanNet 单扫描与 TRAIN 两扫描；CAL/SEL 冻结选模；PB/LOCAL/ADDITIONAL 只用于锁定后确认；D0 保持 lag1/mean。

## 4. 确认的 R1 初始化、最终模型/头的哈希、所需基础权重

R1 `{_fmt(identity.get('r1_checkpoint_sha256'))}`；Concerto `{_fmt(identity.get('concerto_sha256'))}`；最终 recipe `{_fmt(recipe)}`。基础权重由 `external:assets.local.json` 解析，不随 Git 分发。

## 5. 数据 population、物理 reference 数量与历史暴露

TRAIN/CAL/SEL/PB/LOCAL-T2/ADDITIONAL 数量分别为 `{len(role_values.get('TRAIN', ()))}/{len(role_values.get('CAL', ()))}/{len(role_values.get('SEL', ()))}/{len(role_values.get('PB', ()))}/{len(role_values.get('LOCAL-T2', ()))}/{len(role_values.get('ADDITIONAL', ()))}`。PB 是历史暴露人口；LOCAL-T2 与 ADDITIONAL 分表报告。

## 6. 每个臂实际步数、checkpoint 选择和淘汰理由

以 `RUN_STATE.json` 的 `perception_pilot`、`perception_full`、`perception_select` 字段及 `selection/` 冻结文件为准。`NOT_RUN` 与外部阻塞不记为模型实验失败。

## 7. 最终锁定 recipe 与默认部署回退决定

锁定 recipe：`{_fmt(recipe)}`。默认部署由 `confirmation/CONFIRMATION_SUMMARY.json` 的严格 PB 规则决定；无完整确认时保持 `UNCONFIRMED`，不从 PB 反向选模。

## 8. LOCAL/PB/ADDITIONAL 结果、相对 R1 与 C0 的差值

确认状态 `{confirmation_status}`。完整数值和逐 reference 行见 `FINAL_REPORT.md` 与 `confirmation/CONFIRMATION_SUMMARY.json`。

## 9. 当前 native FH 与真实 profile 完成程度

Profile 状态 `{profile_status}`；方法、6 个 canonical masters、1 次整序列 warmup、3 次整序列测量及 scope 见 `resources/PROFILE_SUMMARY.json`。缓存重放不替代 live profile。

## 10. 测试命令与实际结果；没有运行的部分

目标测试、最终回归和 lint 结果以发布前最后一条执行日志及最终报告为准。未执行的训练、确认、profile 会在表 1 和阻塞列表保持显式。

## 11. 真实复现命令、环境、资产解析方法

环境命令前缀 `conda run -n persist4d`；公共执行入口为 `python -m scripts.perception_gain_campaign run --config configs/perception_gain_v1.yaml --external-root "$PERSIST4D_PERCEPTION_RUN_ROOT" --through publish --resume`。实际命令：

{chr(10).join(command_lines)}

## 12. 大文件 release 链接、manifest、取得基础权重的方法

Git 小文件清单见 `ARTIFACT_MANIFEST.json`，拟上传大文件见 `RELEASE_PLAN.json`。基础 R1/Concerto 仅记录 SHA256 和资产键；不重新分发第三方权重或原始数据。

## 13. 剩余阻塞：已执行命令、异常、最小下一动作

{chr(10).join(failure_lines)}

最小下一动作是恢复对应外部依赖后，原命令加 `--resume` 继续，不重置 schedule。

## 14. GitHub 发布验证方法与 release receipt 位置

发布阶段比较本地 HEAD、远端分支 SHA 与 tag 目标；再读取远端 `HANDOFF.md` 和确认 summary。外部 receipt 固定为 `external:publication/PUBLICATION_RECEIPT.json`。当前 publication phase 为 `{publication_phase}`。
"""


def _delivery_paths(project_root: Path, artifact_root: Path) -> list[Path]:
    excluded = {
        artifact_root / "ARTIFACT_MANIFEST.json",
        artifact_root / "RUN_STATE.json",
        artifact_root / "EXECUTION_LOG.jsonl",
        artifact_root / "PUBLICATION_RECEIPT.json",
    }
    candidates = [path for path in artifact_root.rglob("*") if path.is_file()]
    if (project_root / ".git").exists():
        command = [
            "git",
            "diff",
            "--name-only",
            "6ef77620aa20926311eff3124a794a6ca2e32727",
            "--",
            "configs",
            "datasets",
            "models",
            "scripts",
            "tests",
            "trainer",
        ]
        result = subprocess.run(
            command, cwd=project_root, check=False, capture_output=True, text=True
        )
        if result.returncode == 0:
            candidates.extend(
                project_root / value
                for value in result.stdout.splitlines()
                if value and (project_root / value).is_file()
            )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if untracked.returncode == 0:
            candidates.extend(
                project_root / value
                for value in untracked.stdout.splitlines()
                if value
                and (project_root / value).is_file()
                and value.startswith(
                    (
                        "configs/",
                        "datasets/",
                        "models/",
                        "scripts/",
                        "tests/",
                        "trainer/",
                    )
                )
            )
    unique = sorted(
        {path.resolve() for path in candidates if path.resolve() not in excluded}
    )
    return unique


def _manifest(project_root: Path, artifact_root: Path) -> dict[str, object]:
    files = []
    for path in _delivery_paths(project_root, artifact_root):
        size = path.stat().st_size
        if size > 20 * 1024 * 1024:
            raise ReportingError(f"Git delivery file exceeds 20 MiB: {path}")
        files.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "bytes": size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": "perception-gain-artifact-manifest-v1",
        "scope": "Git delivery snapshot; excludes mutable RUN_STATE/EXECUTION_LOG and this manifest",
        "files": files,
    }


def _release_plan(
    *,
    artifact_root: Path,
    external_root: Path,
    experiment_commit: str,
    release_tag: str,
) -> dict[str, object]:
    lock = _read_json(artifact_root / "selection/FINAL_LOCK.json", required=False)
    recipe = lock.get("recipe") if isinstance(lock, Mapping) else {}
    assets = []
    for role, reference, digest in (
        (
            "selected_parent",
            recipe.get("parent_checkpoint") if isinstance(recipe, Mapping) else None,
            (
                recipe.get("parent_checkpoint_sha256")
                if isinstance(recipe, Mapping)
                else None
            ),
        ),
        (
            "selected_refiner",
            recipe.get("refiner_checkpoint") if isinstance(recipe, Mapping) else None,
            (
                recipe.get("refiner_checkpoint_sha256")
                if isinstance(recipe, Mapping)
                else None
            ),
        ),
    ):
        if not isinstance(reference, str) or reference == "external:r1_checkpoint":
            continue
        relative = reference.removeprefix("external:")
        path = external_root / relative
        assets.append(
            {
                "role": role,
                "logical_reference": reference,
                "planned_name": f"{role}-{Path(relative).name}",
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": _sha256(path) if path.is_file() else digest,
                "status": "READY" if path.is_file() else "UNAVAILABLE",
            }
        )
    return {
        "schema_version": "perception-gain-release-plan-v1",
        "tag": release_tag,
        "prerelease": True,
        "experiment_commit": experiment_commit,
        "maximum_part_bytes": 1024**3,
        "assets": assets,
        "receipt": "external:publication/PUBLICATION_RECEIPT.json",
    }


def generate_delivery(
    *,
    project_root: Path = PROJECT_ROOT,
    artifact_root: Path = ARTIFACT_ROOT,
    external_root: Path = DEFAULT_EXTERNAL_ROOT,
    experiment_commit: str,
    publication_phase: str = "READY",
    release_tag: str = "persist4d-perception-gain-v1",
) -> dict[str, object]:
    if len(experiment_commit) != 40 or publication_phase not in {"READY", "PREPARE"}:
        raise ReportingError("report publication identity differs")
    state = _read_json(artifact_root / "RUN_STATE.json")
    run_config = _read_json(artifact_root / "RUN_CONFIG.json")
    roles = _read_json(artifact_root / "DATA_ROLES.json")
    confirmation = _confirmation(artifact_root)
    replication = _read_json(
        artifact_root / "confirmation/REPLICATION_SUMMARY.json", required=False
    )
    profile = _read_json(
        artifact_root / "resources/PROFILE_SUMMARY.json", required=False
    )
    report = _render_report(
        state=state,
        run_config=run_config,
        confirmation=confirmation,
        replication=replication,
        profile=profile,
        artifact_root=artifact_root,
        experiment_commit=experiment_commit,
        publication_phase=publication_phase,
    )
    handoff = _render_handoff(
        state=state,
        run_config=run_config,
        roles=roles,
        confirmation=confirmation,
        profile=profile,
        artifact_root=artifact_root,
        experiment_commit=experiment_commit,
        publication_phase=publication_phase,
    )
    _atomic_write(artifact_root / "FINAL_REPORT.md", report)
    _atomic_write(artifact_root / "HANDOFF.md", handoff)
    _atomic_json(
        artifact_root / "RELEASE_PLAN.json",
        _release_plan(
            artifact_root=artifact_root,
            external_root=external_root,
            experiment_commit=experiment_commit,
            release_tag=release_tag,
        ),
    )
    _atomic_json(
        artifact_root / "ARTIFACT_MANIFEST.json",
        _manifest(project_root, artifact_root),
    )
    return {
        "schema_version": "perception-gain-reporting-v1",
        "status": "PASS",
        "execution_status": _execution_status(state),
        "experiment_commit": experiment_commit,
        "publication_phase": publication_phase,
        "outputs": [
            "FINAL_REPORT.md",
            "HANDOFF.md",
            "ARTIFACT_MANIFEST.json",
            "RELEASE_PLAN.json",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--experiment-commit", required=True)
    parser.add_argument(
        "--publication-phase", choices=("PREPARE", "READY"), default="READY"
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = generate_delivery(
        artifact_root=args.artifact_root.resolve(),
        external_root=args.external_root.resolve(),
        experiment_commit=args.experiment_commit,
        publication_phase=args.publication_phase,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
