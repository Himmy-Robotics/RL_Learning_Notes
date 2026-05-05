# 9. Policy Gradient Methods

<u>本章是从 </u>​**<u>基于价值（value-based）</u>**​**<u>$\to$**</u>​<u> </u>​**<u>基于策略（policy-based）</u>**​<u>  的关键转变。之前所有章节都是先估计价值、再改进策略。本章直接将策略参数化为函数 </u>​<u>$\pi(a|s, \theta)$</u>​<u>，通过优化标量指标 </u>​<u>$J(\theta)$</u>​<u> 来直接搜索最优策略。</u>

‍

## 9.1. 优化指标

**指标1：平均状态价值：**

$\bar{v}_\pi = \sum_{s \in \mathcal{S}} d(s) v_\pi(s)$

其中 $d(s)$ 是状态权重分布。可以选为与策略无关的 $d_0$（如均匀分布），也可以选为策略相关的平稳分布 $d_\pi$。

**指标2：平均奖励：**

$\bar{r}_\pi = \sum_{s \in \mathcal{S}} d_\pi(s) r_\pi(s)$

其中 $r_\pi(s) = \sum_a \pi(a|s, \theta) r(s, a)$。

**等价表达：**

$J(\theta) = \lim_{n \to \infty} \frac{1}{n} \mathbb{E} \left[ \sum_{t=0}^{n-1} R_{t+1} \right] = \bar{r}_\pi$

​**两个指标在折扣情况下等价**：$\bar{r}_\pi = (1 - \gamma) \bar{v}_\pi$，可以同时最大化。

---

## 9.2. 策略梯度定理

$$
\nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s, \theta) q_\pi(s, a)
$$

更实用的​**期望形式**（利用 $\nabla_\theta \pi = \pi \cdot \nabla_\theta \ln \pi$）：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{S \sim \eta, A \sim \pi} \left[ \nabla_\theta \ln \pi(A|S, \theta) q_\pi(S, A) \right]
$$

其中 $\eta$ 是某种状态分布（具体取决于指标和折扣/非折扣情况），$q_\pi$ 是动作价值。

**关键点：**

- 梯度中出现 $\ln \pi$ 是为了将梯度表达为**期望形式**，从而可以用样本近似。
- $\pi(a|s, \theta) > 0$ 是必须的（softmax 保证）。
- 不同指标和场景下，$J$ 和 $\eta$ 不同，但梯度表达式结构相同。

---

## 9.3. REINFORCE算法（蒙特卡洛策略梯度）

用 SGD 替代期望，用 MC 估计的回报替代 $q_\pi$。

**参数更新：**

$$
\theta_{t+1} = \theta_t + \alpha \nabla_\theta \ln \pi(a_t | s_t, \theta_t) q_t(s_t, a_t)
$$

其中 $q_t(s_t, a_t) = \sum_{k=t+1}^{T} \gamma^{k-t-1} r_k$ 是从 $(s_t, a_t)$ 出发的实际折扣回报。

**算法流程：**

1. 用当前策略 $\pi(\theta)$ 生成一整个 episode。
2. 对 episode 中每一步 $t$，计算折扣回报 $q_t$。
3. 更新参数：$\theta \leftarrow \theta + \alpha \nabla_\theta \ln \pi(a_t | s_t, \theta) q_t(s_t, a_t)$

​**直观理解**（改写为 $\theta_{t+1} = \theta_t + \alpha \beta_t \nabla_\theta \pi(a_t | s_t, \theta_t)$，其中 $\beta_t = q_t / \pi(a_t | s_t, \theta_t)$）：

- **$\beta_t$** **正比于** **$q_t$**：动作价值高 $\to$ 增大该动作的概率（​**利用**）。
- **$\beta_t$** **反比于** **$\pi(a_t | s_t, \theta_t)$**​：概率低的动作被增强更多（​**探索**）。
- $q_t < 0$ 时，减小该动作的概率。

💡**REINFORCE 是 on-policy 的，** 样本必须由当前策略生成。

---

‍

‍
