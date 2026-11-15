# 报名邀请试点中不依从与弱推动下的学习收益分析：业务资料契约

households 每户恰有一次随机邀请，概率由 stratum 的 invite_probability 给出；people 关联家庭，attended 是是否参加目标课程，outcome 是标准化社会交往分数，null 为未回访而非零。invitation 的直接影响和 defier 比例是敏感性输入，不是数据识别出的事实。默认只允许在既有随机邀请设计覆盖的人群解释结果；家庭成员及其反事实不能跨情景重新抽签。

所有示例记录由本地确定性生成，不含真实居民、客户或学校个人资料。`revision` 指向整包不可变资料，`synthetic=true` 只说明资料性质，不声明统计结论。

## 当前产品边界

现有分析 `invitation_balance` 已可通过资料目录服务、分析服务和结果回读完整运行；它只给已观察记录的业务核对，不实现 `participation_effect`。新分析需要与当前核对功能并存，统计方法、模型分解和诊断组织由实现者根据证据选择。

新分析的最小产品交付是可保存的结构化分析结果，包含目标人群/估计对象、所用资料及排除原因、估计或可识别范围、适用条件、诊断和业务解释；无法估计时给原因而不是伪造数值。`options` 应允许调用者保存并重放题面涉及的假设或情景，具体字段可随实现补充文档。默认调用用资料能够支持的基本情景，不能把未声明的强假设隐藏进默认值。数值容差、随机种子和蒙特卡洛精度要可复查。

邀请设计在 randomization 中声明：家庭内成员共享处理，户间在各层内独立 Bernoulli 抽取；边际概率不是固定总额无放回设计。

## `participation_effect` 的实现说明

`app/estimands.py` 实现新分析，与 `invitation_balance` 并存。结果同时给出两个互不替代的估计目标：

- `invitation_effect`：被邀请对结局的平均效应，由家庭随机化识别，按层内户数加权合并；它不是参与效应。
- `participation_effect`：真正上课对结局的效应。它依赖依从结构，只给识别界与抽样区间，并附 `baseline`（基础情景）与 `scenarios`（声明情景表），不输出两个均值差相除的点估计。

估计单位为居民，抽样与重抽样单位为家庭：户内成员共享处理，从不作为独立样本；反事实也不在户内重新抽签。结局缺失保留为缺失，用结局量表的上下端给出最坏情形界，绝不以零填补。

`options` 字段（全部在结果 `assumptions.declared` 中回显，便于重放）：

| 字段 | 含义 | 默认 |
| --- | --- | --- |
| `defier_rate` | 基础情景声明的反向依从比例，不是数据识别值 | `0.0` |
| `defier_grid` | 情景表覆盖的反向依从比例集合 | `[0, 0.02, 0.05, 0.10, 0.20]` |
| `social_spillover` | 基础情景中，邀请经由社交而非上课产生的效应占观察总效应的比例 | `0.0` |
| `spillover_grid` | 情景表覆盖的社交比例集合 | `[0, 0.25, 0.5, 1.0]` |
| `spillover_direction` | 社交效应与总效应同向（`same`）或反向（`opposite`） | `"same"` |
| `outcome_bounds` | 缺失结局的取值区间 `[min, max]`；必须包含全部已观察值 | 已观察结局的最小/最大值 |
| `seed` | 家庭层整群 bootstrap 的随机种子 | `20261115` |
| `bootstrap_replicates` | bootstrap 重复次数 | `2000` |
| `confidence` | 区间置信水平 | `0.95` |

未知字段直接 422，避免把未声明假设悄悄带入默认结果。情景表每行给出 `identification_bounds`（识别界）、`sampling_interval`（把识别界与家庭抽样不确定性合并后的区间）、`complier_share`、`direct_effect_component`、`status` 和 `conclusion`。识别界与抽样区间分开记录：前者来自缺失与依从假设，后者来自家庭重抽样。

当邀请没有推动报名时，`first_stage` 为 0，参与效应为 `not_identified`，不生成比值；当基础情景声明了不可行的反向依从（超过对照组参与比例）时，该行标为 `infeasible_defiers`。情景汇总 `summary.direction` 只在所有可行情景同号时才给方向，否则明确为 `sign_not_resolved`。

`source` 记录所用的 `revision`、家庭与人数、分层、被排除的无成员家庭，以及缺失回访计数；`precision` 记录方法、种子、重复次数与置信水平。保存的 envelope 另含 `source_sha256` 与 `analysis`，可从结果回读到分析对象和依据。


## 已有业务投影视图

`app/design.py` 的 `design_profile` 为当前核对分析生成 `source_profile`，保留此项目的分析单位、重复观测和资料组织。这些已观察资料的投影不是新分析结果；可复用其数据关系，但不能把描述统计当成推断。修改现有资料处理时要保留 `tests/test_context.py` 的业务不变量。
