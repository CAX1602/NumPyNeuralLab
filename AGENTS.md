# AGENTS.md — NumPy Neural Lab · 游戏前端任务源（DSH 接手）

> 本文件是 **DSH（DeepSeek Harness）** 在本目录工作时的任务边界与规则源；Qoder 会话同样遵守此分工。
> 分工：**DSH 负责游戏前端**——`game.py`、`characters.py`、两个预览脚本、游戏相关画面、实玩手感修复。
> **Qoder 负责网络与物理核心**——`network.py`、`physics.py`、`data.py`、`train.py`、`experiment*.py`、`gradients.py`、`tests/`、`make_shots.py`、报告与排期。
> 权威进展：本目录 `进展记录.md`（设计规则全文在第九节，游戏现状在第十二节）。
> 跨环境状态入口：桌面 `NumPyNeuralLab-游戏前端-交接.md`。续做先读这两份，再读本文件。

## 0. 交接时状态（2026-11-16）

- 游戏骨架已完成并自测：35/35 单元测试全绿；`--smoke` 双 AI 30 s：60.4 fps、6 回合 54 次击球（≈9 次/回合，连续多拍达成）。
- 已修两个实玩问题：① 方向键 ←/→/↑ 映射（用户习惯方向键）；② 挥拍动画（`characters.py` 的 `racket_theta` 参数，拍面随 θ 旋转 + 白色残影弧，渲染目检过）。
- **未 commit 批**：`game.py`（新增）、`characters.py`（修改）、`进展记录.md`（修改）。等用户发话才 commit + push，见第 4 节。
- 待办（DSH 接手）：用户 PyCharm 实玩手感验收（11-21 周六节点）——挥拍时机、AI 强度、跳跃拦截；AI 起跳拦截与快球 WAIT 提前量试玩调。

## 1. 文件边界

- **可改**：`game.py`、`characters.py`、`game_style_preview.py`、`character_select_preview.py`、`docs/` 游戏相关图、`进展记录.md`、桌面交接文件。
- **不可改**：`network.py`、`physics.py`、`data.py`、`train.py`、`experiment*.py`、`gradients.py`、`tests/`、`make_shots.py`、`shots.npz`、`landing.py`、`landing_baseline.py`。需要动物理或网络 → 停下来告诉用户转 Qoder。
- **接口契约**：`game.py:156` `AIController.predict(self, ball)` 保持签名与语义（返回落点 x）。现在内部是模拟器占位；**11-28 由 Qoder 换成落点网络**，不要重构掉这个接口。

## 2. 固定设计规则（用户已定，不许回退）

- 验收基线 = 早期 2H5 版 4399《火柴人打羽毛球》；只关心「用户电脑跑得动 + 好不好玩」，增加操作复杂度或性能开销的细节默认砍。
- **接到球第一**，进攻后谈；先到 7 分；12 角色（speed/power/jump）全部可跳；AI 用镜像角色（i+6）。
- 单键 **空格** 挥拍；A/D 或 ←/→ 移动，W 或 ↑ 跳。按下瞬间球低于腰部 → 挑球，否则高远；`waist_y = feet_y - 90*scale`，**随跳跃上移**。
- 两种固定挥拍弧：clear θ 1.20→−0.55 / 0.50 s；lift θ 1.05→−0.15 / 0.32 s。接触帧读 θ 与 s；**出射角只看 θ**；`v0 = v_base(θ) × (0.6+0.4s) × power`。
- 无风，但**空气阻力必须可见**：二次阻力 k=0.0025，与 `physics.py` 同方程、240 Hz 子步（dt=1/240）。
- 接触判定 = 球到拍杆线段（手→拍头）距离 < 34 px，s 为投影；不能连击自己刚打出的球。
- AI 状态机 RECOVER→WAIT→CHASE→SWING；**球已过网或快球允许从 RECOVER/WAIT 直接 CHASE**（踩过的坑：走完回位才追永远接不到）。已定标：起挥距离 120、站位偏移 25。
- **物理结论更新（11-19）**：早期「二次阻力下最大射程 ~345 px < 半场 450，深区必不过网」是**原速（380/820）**下的发现；11-17 起多次提速、11-19 已到 `V_MIN/V_MAX = 1520/3280`（4 倍），深区也能过网，旧结论作废。不要改物理去「修」射程；AI 回位深度只作为手感调参，现状以 `进展记录.md` 第十二节为准。

## 3. 运行与验证（本机实测过的命令）

- 一律用项目 venv：`.venv/Scripts/python.exe`。**全局 D:\Python 没有 pygame**，用错解释器会 ModuleNotFoundError。
- 测试：`.venv/Scripts/python.exe -m unittest discover -s tests`（60 例，应全绿；含落点网络与线性基线，全部只用合成数据，不训练真实数据）。
- 自动对打：`.venv/Scripts/python.exe game.py --smoke`（30 s，看 fps 与 hits/rally；基线 60.4 fps、9 次/回合）。
- 画面改动：pygame 可无窗口渲染 `Surface` 存 png 到 `docs/`，然后 Read 目检（挥拍动画就是这么验的）。
- 实玩由**用户在 PyCharm 跑 `game.py` 并贴反馈**；你负责改代码与自测，不要替用户「玩」。
- 中文输入法会吞字母键：方向键已映射；若空格失灵，提醒用户切英文输入。
- `characters.py` 的 `import pygame` 在 `draw_character` 内部（懒导入），数据层可脱离 pygame——保持这个结构。

## 4. git 约定

- **永不更新 git config**（身份已配好，仓库本地）。
- **commit 必须用户发话**；当前未 commit 批见第 0 节，不得 reset/checkout 丢弃。
- 推送走 Clash 代理：`HTTPS_PROXY=http://127.0.0.1:7897 git push`（直连间歇被断）。
- 远程：https://github.com/CAX1602/NumPyNeuralLab（private，main）。credential helper 已记住 manager；若 push「卡死」，先问用户桌面有无弹窗等点击，别当网络问题干等。

## 5. 排期与协作

- 11-21（周六）：游戏手感验收节点（DSH 主责）。11-27/11-28：落点网络训练与接入（Qoder 主责，DSH 保持 predict() 接口稳定）。12-05：三 AI 对照 + 熔断决策（Qoder 主责）。
- 游戏只准占用 11-21 / 11-28 / 12-05 三个周六，不许吃 12-12 之后的测试与交付时间。
- 用户是北大计概大一学生：解释先类比/图、配置给点击步骤；代码你写你验，用户跑并贴输出。

## 6. 每次对话结束

更新 `进展记录.md`（头部实时时间、第十二节、更新日志三处），并同步桌面交接文件的状态与头部时间。