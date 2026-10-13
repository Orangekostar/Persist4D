# MPR-V1 执行设计

权威规范为 `artifacts/revisit_prompt_v1/EXECUTION_INSTRUCTION.md`（SHA256
`08cf2d2c83f1bd93d3120623f3ea9884fddd0f1fb13af843677b92160cc0070e`）。
用户已要求自主执行并推断推荐配置，因此不增加交互批准步骤。

这是架构工作：新增首掩码前提示路径和受限适配训练、评价、发布入口。
沿用原生 E1-23760 的点云、编码器、解码器、损失和官方候选处理。
首选方案为薄适配层；重新实现分割器会破坏父权重和指标契约，继续旧读取
则无法测试首掩码前假设。新系统只改 ReScene 的空 hook 和可选冻结上下文入口。

## 模型及状态

`RevisitPromptReScene` 接收 label-free StageMeta、TaskMemoryState 和几何 sidecar。
当前段特征按真实 XYZ 和段内局部时间严格对齐。先从预测旧状态选最多32槽，
M4–M6全局检索当前段，再以32近邻聚合，更新查询内容和当前位置。
M1用当前FPS位置附近的当前特征替代历史内容；M3只用历史内容与旧位置。
M2重用 TaskMemoryRead 的原插入点并限制已选槽；M0无提示参数。
所有参数集合、初始化、拒绝评分和辅助损失严格使用规范§7–10。

所有变体仅按最终预测重新 route/commit 一次，ID来自 CommitResult。
几何更新只接受当前有效且提交成功的预测。历史标签账本是训练侧对象，
不进入模型。空历史严格不改父查询。父参数与缓冲区严格覆盖，父模块eval，
梯度通过冻结解码器回传新增参数。

## 数据与指标

UUID spec 保留数据索引scope；TRAIN/CAL/SEL是train索引，PB是validation索引。
每输入从空状态因果重放，首阶段单扫描，后续只读上一扫描和当前扫描。
每步四episode长度依次2/3/4/5；每阶段backward后detach，truncation=1。
GT使用同一完整因果前缀的类别语义投影，仅供损失与评价。
官方候选保留source_query_ids和ephemeral轨迹；commit0，分数mean。
M6在官方提取后相对原分数缩放一次，主损失仍用原logits。
指标状态按总体池化；三个新进程重复各自计分，不能拼接重复。

## 运行、选择与交付

预算绑定旧总账1800、旧消费1713.8830547291614和8小时预留；新支出最多78。
单GPU、既有persist4d环境，成本按整进程计费；不建设DDP队列。
P0/P1实测后一次性锁定P2–P5与N∈{400,800,1200,1600}。
所有六臂和最多两臂seed147完整预测必须能覆盖，否则不启动偏置矩阵。
CPU测试仅规范T01–T10；真实接通和重复面板计入预算。
选择、失败分支、seed147晋级、SEL/PB锁定按规范§12–15执行。
真实结果、delta及必要resume、官方状态发布到新branch/tag；A绑定实验，
B交接，最终远端下载核验回执在Release或本地，避免SHA自引用。

## 接口及文件

模型/状态/损失由 `models/revisit_prompt*.py`；数据由
`datasets/revisit_prompt_episode.py`；系统和优化循环由
`trainer/revisit_prompt_trainer.py`。运行、资产、评价、选择、发布与编排
分别用规范§4的同名scripts。所有入口支持help和实际状态输出。
最终产物清单和12项自审以规范§17–19逐项核验，代码、实验、科学、发布
状态独立，不能以测试通过代替科学收益。
