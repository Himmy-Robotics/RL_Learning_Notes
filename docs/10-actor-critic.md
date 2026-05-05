# 10. Actor-Critic Methods

<u>本章介绍Actor-Critic ，Actor-Critic 是 </u>​**<u>策略梯度 (policy-based) + 价值估计 (value-based)</u>**​<u>  的结合。第 9 章的 REINFORCE 用 MC 估计 </u>​<u>$q_\pi$</u>​<u>（方差高），本章改用 TD 估计 </u>​<u>$q_\pi$</u>​<u>（方差低），形成 Actor-Critic 框架。</u>

- **Actor（演员）**  ：策略网络 $\pi(a|s, \theta)$，负责选动作，通过策略梯度更新 $\theta$。
- **Critic（评论家）**  ：价值网络 $q(s, a, w)$ 或 $v(s, w)$，负责评估动作好坏，通过 TD 更新 $w$。

‍

## 10.1. QAC（Q value Actor-Critic）

直接将 REINFORCE 中的 MC 估计替换为 TD 估计。

**Actor 更新**（策略梯度）：

$$
\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \ln \pi(a_t | s_t, \theta_t) q(s_t, a_t, w_t)
$$

​**Critic 更新**（Sarsa + 函数近似）：

$$
w_{t+1} = w_t + \alpha_w \left[ r_{t+1} + \gamma q(s_{t+1}, a_{t+1}, w_t) - q(s_t, a_t, w_t) \right] \nabla_w q(s_t, a_t, w_t)
$$

​**核心思想**：Critic 估计当前策略的动作价值，Actor 根据这个价值来改进策略。两者交替进行。

---

## 10.2. A2C（Advantage Actor-Critic）

**基线不变性：**

策略梯度对任意仅依赖状态的基线 $b(S)$ 不变：

$$
\mathbb{E}[\nabla_\theta \ln \pi(A|S, \theta) q_\pi(S, A)] = \mathbb{E}[\nabla_\theta \ln \pi(A|S, \theta) (q_\pi(S, A) - b(S))]
$$

原因：$\sum_a \nabla_\theta \pi(a|s, \theta) = \nabla_\theta \sum_a \pi(a|s, \theta) = \nabla_\theta 1 = 0$

虽然梯度的**期望**不变，但**方差**会变。选好的基线可以大幅降低方差。

**优势函数：**

选基线 $b(s) = v_\pi(s)$，定义​**优势函数**：

$$
\delta_\pi(s, a) = q_\pi(s, a) - v_\pi(s)
$$

含义：动作 $a$ 相对于平均水平的"优势"。$\delta_\pi > 0$ 说明这个动作比平均好。

实际中用 **TD 误差** 近似优势函数（只需一个 $v$ 网络，不需要 $q$ 网络）：

$$
q_\pi(s_t, a_t) - v_\pi(s_t) \approx r_{t+1} + \gamma v(s_{t+1}, w_t) - v(s_t, w_t) = \delta_t
$$

**A2C 流程：**

1. **Advantage（TD 误差）**：

	$$
	\delta_t = r_{t+1} + \gamma v(s_{t+1}, w_t) - v(s_t, w_t)
	$$

2. **Actor 更新**：

	$$
		heta_{t+1} = \theta_t + \alpha_\theta \delta_t \nabla_\theta \ln \pi(a_t | s_t, \theta_t)
	$$

3. **Critic 更新**：

	$$
	w_{t+1} = w_t + \alpha_w \delta_t \nabla_w v(s_t, w_t)
	$$

​**优势**：只需维护一个价值网络 $v(s, w)$，不需要 $q$ 网络。策略 $\pi(\theta)$ 本身是随机的、探索性的，不需要 $\epsilon\text{-greedy}$。

---

## 10.3. Off-policy Actor-Critic 

### 10.3.1. **重要性采样：**

目标：用分布 $p_1$ 生成的样本来估计分布 $p_0$ 下的期望。

$$
\mathbb{E}_{X \sim p_0}[X] = \mathbb{E}_{X \sim p_1} \left[ \frac{p_0(X)}{p_1(X)} X \right] \approx \frac{1}{n} \sum_{i=1}^{n} \frac{p_0(x_i)}{p_1(x_i)} x_i
$$

其中 $\frac{p_0(x_i)}{p_1(x_i)}$ 是**重要性权重**。

### 10.3.2. **Off-policy 策略梯度定理：**

用行为策略 $\beta$ 的样本来优化目标策略 $\pi$：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{S \sim \rho, A \sim \beta} \left[ \frac{\pi(A|S, \theta)}{\beta(A|S)} \nabla_\theta \ln \pi(A|S, \theta) q_\pi(S, A) \right]
$$

与 on-policy 的区别：多了重要性权重 $\frac{\pi}{\beta}$，动作从 $\beta$ 而非 $\pi$ 采样。

💡**我们使用的数据是由旧策略收集到的，而我们想要用这些数据来优化新策略，因此需要给旧策略得到的数据前面加一个权重，也就是重要性采样权重，这样旧策略获取到的数据就可以用于新策略的更新了。**

### 10.3.3. **Off-policy Actor-Critic 算法：**

在 A2C 的基础上，Actor 和 Critic 都乘以重要性权重：

$$
\theta_{t+1} = \theta_t + \alpha_\theta \frac{\pi(a_t|s_t, \theta_t)}{\beta(a_t|s_t)} \delta_t \nabla_\theta \ln \pi(a_t | s_t, \theta_t)
$$

$$
w_{t+1} = w_t + \alpha_w \frac{\pi(a_t|s_t, \theta_t)}{\beta(a_t|s_t)} \delta_t \nabla_w v(s_t, w_t)
$$

---

## 10.4. Deterministic Actor-Critic (DPG——Deterministic Policy Gradient)

### 10.4.1. **确定性策略：**

用 $a = \mu(s, \theta)$ 表示确定性策略（不是概率分布，直接输出动作）。

### 10.4.2. **确定性策略梯度定理：**

$$
\nabla_\theta J(\theta) = \mathbb{E}_{S \sim \eta} \left[ \nabla_\theta \mu(S) (\nabla_a q_\mu(S, a)) |_{a = \mu(S)} \right]
$$

**关键区别：** 梯度中没有动作随机变量 $A$，所以不需要对动作采样 $\to$ ​**天然 off-policy**。

**直观含义：**

- **$\nabla_a q_\mu(s, a) |_{a = \mu(s)}$**：在当前动作处，$q$ 值关于动作的梯度方向（“动作应该往哪个方向调整能提高 $q$ 值”）。
- **$\nabla_\theta \mu(s)$**：参数 $\theta$ 往哪个方向变能改变动作输出。

两者的链式组合告诉 $\theta$ 应该怎么更新。

### 10.4.3. **确定性 Actor-Critic（DPG）：**

​**Actor**：

$$
\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu(s_t, \theta_t) (\nabla_a q(s_t, a, w_t)) |_{a = \mu(s_t)}
$$

​**Critic**（Sarsa-like，但用目标策略 $\mu$ 生成 $\tilde{a}_{t+1}$）：

$$
\delta_t = r_{t+1} + \gamma q(s_{t+1}, \mu(s_{t+1}, \theta_t), w_t) - q(s_t, a_t, w_t)
$$

$$
w_{t+1} = w_t + \alpha_w \delta_t \nabla_w q(s_t, a_t, w_t)
$$

💡**Off-policy 但不需要重要性权重，** 因为梯度中不涉及动作采样。行为策略 $\beta$ 可以是任意探索性策略。DDPG 是其深度学习版本。

---

‍

‍
