# 正式训练运行命令

使用 assets.local.json 中已绑定的 Python 和环境，在本分支工作区执行。

```bash
python -m scripts.continue_revisit_prompt_formal
```

当前已有独立后台控制器，进程锁拒绝并行重复启动。训练和评测由原收费管理器执行。

读取实时训练状态：

```bash
python - <<'PYCODE'
import json
from pathlib import Path
a=json.loads(Path("assets.local.json").read_text())
r=Path(a["run_root"])
print((r/"CONTROLLER_STATUS.json").read_text())
for p in sorted((r/"trajectories").glob("*/STATUS.json")):
    s=json.loads(p.read_text())
    print(p.parent.name,s["status"],s["step"],s["planned_updates"])
PYCODE
```

活动锁不可手工删除；已有正式轨迹通过控制器和原 --resume 路径恢复。resolve 命令使用旧协议默认预算，不用于本次已绑定预算修订。发布原始协议的脚本不自动执行；正式结果完成后由主执行者复核。
