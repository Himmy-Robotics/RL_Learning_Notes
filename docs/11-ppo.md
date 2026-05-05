# 11. PPO（Proximal Policy Optimization）

本章介绍PPO，PPO要解决的问题说到底就是使用了重要性采样使得Policy Gradient由on-policy变为off-policy的时候，用于sample的policy的参数和agent的policy的参数不能相差太大，因此通过引入KL散度，为了让两个policy的参数尽可能接近，通过引入KL散度的方法的不同又可以分为TRPO，PPO-Penalty，PPO-Clip。

## 11.1. KL Divergence

**直观理解:** 刻画两种概率分布之间的“差距”。如果真实分布是$p$​$﻿$ ，但你错误地以为分布是$q$ ，你会损失多少信息。

**定义：** 假设有两个概率分布 $p$ 和 $q$，定义在同一个集合上。KL 散度为：

$$
D_{KL}(p \| q) = \sum_{x} p(x) \ln \frac{p(x)}{q(x)}
$$

如果是连续分布，求和换成积分：

$$
D_{KL}(p \| q) = \int p(x) \ln \frac{p(x)}{q(x)} dx
$$

当两个分布完全相同时$D_{KL}=0$。

---

## 11.2. GAE（Generalized Advantage Estimation）

一种用于求Advantage的方法，在MC方法和TD方法中的trade-off。

**优势函数的定义：**

$$
A_\pi(s_t, a_t) = q_\pi(s_t, a_t) - v_\pi(s_t)
$$

1. **用TD方法求（A2C的做法）：**

$$
\hat{A}_t^{(1)} = \delta_t = r_{t+1} + \gamma v(s_{t+1}) - v(s_t)
$$

**偏差大**，严重依赖当前价值函数 $v(s, w)$ 的准确性。如果 $v$ 估得不对，优势函数步步皆错。

2. **用完整的MC回报：**

$$
\hat{A}_t^{(\infty)} = \left( \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \right) - v(s_t) = G_t - v(s_t)
$$

**方差高**，涉及整个 Episode 的所有随机变量，如果环境随机性大或 Episode 很长，方差会爆炸。

3. **GAE方法：**

$$
\hat{A}_t^{GAE} = \delta_t + \gamma \lambda \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \cdots = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

💡其意义在于保证优势的数值准确的同时稳定。

---

## 11.3. TRPO (Trust Region Policy Optimization)

​**TRPO 目标函数**：

$$
J_{TRPO}^{\theta'}(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \frac{p_\theta(a_t | s_t)}{p_{\theta'}(a_t | s_t)} A^{\theta'}(s_t, a_t) \right]
$$

**约束条件（置信区域）** ：

$$
KL(\theta, \theta') \le \delta
$$

---

## 11.4. PPO-Penalty

**初始化**：

- 初始化策略参数 $\theta^0$。

**每一轮迭代**：

- 使用当前策略 $\theta^k$ 与环境交互，收集轨迹数据 $\{s_t, a_t\}$ 并计算优势函数 $A^{\theta^k}(s_t, a_t)$。
- 寻找最优的 $\theta$ 来最大化 $J_{PPO}(\theta)$。

​**核心目标函数**：

$$
J_{PPO}^{\theta^k}(\theta) = J^{\theta^k}(\theta) - \beta KL(\theta, \theta^k)
$$

**替代目标函数 (Surrogate Objective)**  ：

$$
J^{\theta^k}(\theta) \approx \sum_{(s_t, a_t)} \frac{p_\theta(a_t | s_t)}{p_{\theta^k}(a_t | s_t)} A^{\theta^k}(s_t, a_t)
$$

**自适应 KL 惩罚系数更新**：

- 如果 $KL(\theta, \theta^k) > KL_{max}$，增大 $\beta$（惩罚更严厉，减小步长）。
- 如果 $KL(\theta, \theta^k) < KL_{min}$，减小 $\beta$（放宽限制，允许更大步长）。

---

## 11.5. PPO-Clip

与PPO-Penalty的区别为将目标函数中的KL散度使用了简单了Clip机制进行刻画：

$$
J_{PPO2}^{\theta^k}(\theta) \approx \sum_{(s_t, a_t)} \min \left( \frac{p_\theta(a_t | s_t)}{p_{\theta^k}(a_t | s_t)} A^{\theta^k}(s_t, a_t), \text{clip} \left( \frac{p_\theta(a_t | s_t)}{p_{\theta^k}(a_t | s_t)}, 1-\epsilon, 1+\epsilon \right) A^{\theta^k}(s_t, a_t) \right)
$$

**PPO-Clip 的 Loss 梯度形式：**

1. **定义：**

	**概率比率：**

	$$
	r_t(\theta)=
	\frac{\pi(a_t|s_t,\theta)}
	{\pi(a_t|s_t,\theta_{old})}
	$$

	**优势函数**：

	$$
	\hat A_t
	$$

2. **Policy Loss：**

	**PPO-Clip 目标函数：**

	$$
	L^{CLIP}(\theta)
	=
	\mathbb{E}_t
	\left[
	\min
	\left(
	r_t(\theta)\hat A_t,\;
		ext{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
	\right)
	\right]
	$$

	**Policy 梯度：**

	由于$\pi(a_t|s_t,\theta_{old})$是常数，因此

	$$
	\nabla_\theta r_t(\theta)
	=
	r_t(\theta)
	\nabla_\theta
	\log \pi(a_t|s_t,\theta)
	$$

	因此：

	$$
	\nabla_\theta L^{CLIP}
	=
	\mathbb{E}_t
	\left[
	g_t
	\hat A_t
	\nabla_\theta
	\log \pi(a_t|s_t,\theta)
	\right]
	$$

	其中：

	$$
	g_t=
	\begin{cases}
	r_t(\theta),
	& \text{if not clipped}
	\\
	0,
	& \text{if clipped}
	\end{cases}
	$$

3. **Critic** **Loss：**

	**价值函数损失：**

	$$
	L^{critic}(\phi)
	=
	\mathbb{E}_t
	\left[
	(V(s_t,\phi)-\hat R_t)^2
	\right]
	$$

	**Critic 梯度：**

	$$
	\nabla_\phi L^{critic}
	=
	\mathbb{E}_t
	\left[
	2
	\big(
	V(s_t,\phi)-\hat R_t
	\big)
	\nabla_\phi
	V(s_t,\phi)
	\right]
	$$

4. **Actor 加上 Entropy：**

	$$
	L^{actor}
	=
	L^{CLIP}
	+
	c_{ent}
	\mathbb{E}_t
	\left[
	H(\pi(\cdot|s_t))
	\right]
	$$

	**梯度：**

	$$
	\nabla_\theta L^{actor}
	=
	\nabla_\theta L^{CLIP}
	+
	c_{ent}
	\nabla_\theta H(\pi)
	$$

5. **最终参数更新：**

	**Actor：**

	$$
		heta
	\leftarrow
		heta
	+
	\alpha_\theta
	\nabla_\theta
	L^{actor}
	$$

	**Critic：**

	$$
	\phi
	\leftarrow
	\phi
	-
	\alpha_\phi
	\nabla_\phi
	L^{critic}
	$$

‍

‍
