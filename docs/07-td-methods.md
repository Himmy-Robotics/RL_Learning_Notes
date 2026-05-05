# 7. Temporal-Difference Methods

<u>本章介绍TD方法，TD方法是典型的</u>​**<u>无模型 + 增量式随机近似</u>**​<u>的强化学习算法。与MC方法相比，TD不需要等整个episode结束才更新，</u>​**<u>每走一步就能立即更新价值估计，TD估计的本质就是RM算法</u>**​<u>。</u>

## 7.1. TD Learning

**TD估计的定义：**

$$
\underbrace{v_{t+1}(s_t)}_{\text{新的估计值}} = \underbrace{v_t(s_t)}_{\text{当前估计值}} - \alpha_t(s_t) \overbrace{[v_t(s_t) - (\underbrace{r_{t+1} + \gamma v_t(s_{t+1})}_{\text{TD 目标}})]}^{\text{TD 误差}},
$$

$$
\underbrace{v_{t+1}(s_t)}_{\text{新的估计值}} = \underbrace{v_t(s_t)}_{\text{当前估计值}} + \alpha_t(s_t) \overbrace{[ \underbrace{r_{t+1} + \gamma v_t(s_{t+1})}_{\text{TD 目标}} - v_t(s_t) ]}^{\text{TD 误差}},
$$

**TD目标**：$\bar{v}_t = r_{t+1} + \gamma v_t(s_{t+1})$，RM算法保证了随着更新次数的增加， $v_t(s_t)$会收敛到目标值

**TD误差：**​$\delta_t = v_t(s_t) - (r_{t+1} + \gamma v_t(s_{t+1}))$，反映当前估计与新样本之间的差异

**与MC方法的对比：**

- TD是增量式的，每步更新；MC是非增量式的，需等episode结束
- TD能处理continuing tasks（无终止状态）；MC只能处理episodic tasks
- TD有bootstrapping（用当前估计更新估计），需要初始猜测；MC不需要
- **TD估计方差低但有偏差；MC估计方差高但无偏**

---

## 7.2. Sarsa

将TD Learning中的估算state value变为的估算action value，每一步需要$(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})。$

**TD估计用于q值估算：**

$$
q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t) [q_t(s_t, a_t) - (r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1}))]
$$

**策略更新：**  Sarsa本身只做策略评估，需结合策略改进。具体做法是：每更新一个q值后，立即将该状态的策略更新为 ϵ-greedy策略，然后用更新后的策略生成下一步样本。这是广义策略迭代的思想。

---

## 7.3. n-step Sarsa

**核心思想：** 回报 $G_t$ 可以按不同的步数分解。

$$
G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^n q_{\pi}(S_{t+n}, A_{t+n})
$$

- 当 $n=1$，TD目标为 $r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})$，就是 Sarsa
- 当 $n=\infty$，TD目标为 $r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots$，就是 MC（MC就是把要把$G_t$给算出来而不涉及到估计）
- 一般 $n$，TD目标为 $r_{t+1} + \gamma r_{t+2} + \dots + \gamma^n q_t(s_{t+n}, a_{t+1})$，就是 n-step Sarsa

**偏差-方差权衡**：$n$ 小时接近 Sarsa，低方差但较大偏差（bootstrapping多）；$n$ 大时接近 MC，高方差但小偏差。

---

## 7.4. Q-learning

Q-learning与Sarsa的区别在于TD target的选择，通过修改TD目标，本质上在求解的是贝尔曼最优公式。

**更新公式：**

$$
q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t) [q_t(s_t, a_t) - (r_{t+1} + \gamma \max_a q_t(s_{t+1}, a))]
$$

**运行逻辑：**

- Q-learning是off-policy的，其行为策略和目标策略不相同，行为策略是一个随机的策略，不会被更新，而目标策略是一个greedy的策略。
- Q-learning的运行逻辑是维系一张Q值表，行为策略决定了每一步的action，目标策略用于更新Q值，每一个新的时间t+1都会更新Q值表，Q值表更新后目标策略就会随之更新，因为目标策略在一个状态所选择的action是根据Q值表来greedy选择的。**由于行为策略具有强随机性，而目标策略是greedy的，因此可以保证当探索的足够多时，目标策略会收敛到最优。**

---

## 7.5. on-policy and off-policy

**on-policy**：行为策略 \= 目标策略（Sarsa、MC）

**off-policy**：行为策略 ≠ 目标策略（Q-learning），优势在于可以用强探索性策略收集数据，甚至是可以利用已有的数据进行学习

‍

‍
