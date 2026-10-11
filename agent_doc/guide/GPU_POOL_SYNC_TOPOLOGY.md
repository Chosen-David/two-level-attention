# GPU 资源池代码同步拓扑（2026-10-11 用户指令确立）

## 双 remote 拓扑

```
                    ┌─── GitHub origin（git@github.com:Chosen-David/two-level-attention.git）
                    │    用途：外部 AI 访问窗口、异地备份
本地 ~/two-level-attention ──┤
                    │    ┌── .187 ─┐
                    └─── BeeGFS bee 裸仓库 ──┼── .24（H20）──┤  ← GPU 机群日常同步面
                         /mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/two-level-attention.git
                         （共享盘文件协议，不经过网络栈）
```

- **BeeGFS bee = 机群 canonical remote**：纯内网文件协议，无外网依赖、无 .24 只读问题、远程可 push 回流 hotfix。
- **GitHub origin = 外部窗口**：给远端 GPT 等外部 AI fetch；双推纪律：`git push bee main && git push origin main`。
- 美团内部 git.sankuai.com 废弃（2026-10-11 用户指令）；.251 旧美团树备份为 `~/two-level-attention_sankuai_bak_20261011`。

## 各节点当前状态（2026-10-11 18:3X 建立）

| 节点 | 树路径 | git 状态 | 备注 |
|---|---|---|---|
| 本机 | ~/two-level-attention | bee+origin 双 remote，main@54c28b6 | 权威工作树 |
| .251 | ~/two-level-attention | bee@54c28b6，已换树 | 双卡空闲，下次派单直接用 |
| .187 | ~/two-level-attention_git | bee@54c28b6，平行目录 | 在飞 E125 三臂仍用旧 rsync 树 ~/two-level-attention（md5 同 54c28b6 关键文件，d993d748/751bcfbc）；链收官后换树 |
| .24 (H20) | ~/two-level-attention_git | bee@54c28b6，平行目录 | 在飞 RULER 尾格仍用旧树；链收官后换树 |

裸仓 HEAD 已修指 refs/heads/main（git init --bare 默认 master 会致 clone 报 nonexistent ref，新裸仓必查）。

## 源身份口径（git 时代）

- 派单身份登记：**git commit SHA（首选）** + 关键文件 md5（兼容旧记录）。
- 派单前核对：`git fetch bee && git log -1` 确认节点 checkout 与预期 SHA 一致。
- 裸仓库带版本/校验和/增量传输，旧 rsync+md5 manifest 纪律全部内化，拉回循环退役（见下）。

## 结果面直写 BeeGFS（下一步，等在飞链收官后启用）

各 runner `--output-dir` 直接写共享盘 results 目录（建议 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/results/<实验代号>/<arm>/），本机打分机直读，rsync 拉回循环（10min 周期各 pull 脚本）整体退役。在飞 E125/E116b 保持现输出目录不动，**新派单起**执行新口径。
