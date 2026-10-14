# MPR 正式训练继续执行

状态：RUNNING。2026-10-14 用户授权修订预算，独立运行，不覆盖旧协议发布。

正式方案：M1–M6，seed146，N=400；四个评测点为 100/200/300/400。条件 seed147 复验和全部 CAL/SEL/PB 规则沿用原实验。

后台控制器：`python -m scripts.continue_revisit_prompt_formal`。活动进程及实时步数以 assets.local.json 对应 run_root 下 CONTROLLER_STATUS.json、ACTIVE_JOB.json 和 trajectories/*/STATUS.json 为准。

本轮新池 100 GPU 小时，累计上限 1823 GPU 小时，历史 1714.814061 GPU 小时完整保留。详细预算与授权见 PROTOCOL_AMENDMENT.json 和 RUN_LOCK.json。

原已发布 BLOCKED 结果保留于 Git commit 0c958f535679a94b39b5661fba0b900305582170。当前科学结论仍为 INCONCLUSIVE；正式训练、评测、部署复核和最终投稿审阅尚未完成。
