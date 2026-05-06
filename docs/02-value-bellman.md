# 2. State-Value, Action-Value and Bellman Equation

<u>本章介绍强化学习中最重要的概念Value，强化学习的过程实际上就是在最大化Value，同时介绍贝尔曼公式。</u>

‍

## 2.1. 核心定义

一切价值的本质都是未来奖励的累积。定义在时间步 $t$ 的**折扣回报（Discounted Return）**  为随机变量 $G_t$：

$$
G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

其中 $\gamma \in [0, 1]$ 是折扣因子。

### 2.1.1.  状态价值函数 (State-Value Function) $v_\pi(s)$

**定义**：在状态 $s$ 下，按照策略 $\pi$ 进行决策，所能获得的**长期回报的数学期望**。

$$
v_\pi(s) \doteq \mathbb{E}_\pi [G_t \mid S_t = s]
$$

### 2.1.2.  动作价值函数 (Action-Value Function) $q_\pi(s, a)$

**定义**：在状态 $s$ 下，**强制执行动作** **$a$**，之后继续按照策略 $\pi$ 进行决策，所能获得的**长期回报的数学期望**。

$$
q_\pi(s, a) \doteq \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]
$$

---

## 2.2. V 与 Q 的相互转化关系与推导

状态价值 $V$ 与动作价值 $Q$ 可以通过**策略** **$\pi(a|s)$** 和**环境转移概率** **$p(s', r \mid s, a)$** 相互转换。

### 2.2.1.  用 Q 表示 V (基于策略的期望)

状态 $s$ 的整体价值，等于在该状态下所有可能动作的 $Q$ 值，按策略概率进行的**加权平均**：

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) q_\pi(s, a)
$$

### 2.2.2.  用 V 表示 Q (基于环境的期望)

**推导过程（利用全期望公式与马尔可夫性）：**

1. 展开 $Q$ 的定义：$q_\pi(s, a) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a]$
2. **引入全期望公式**（按照下一步所有可能的环境转移 $s'$ 和 $r$ 进行展开）：


	$$
	q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \mathbb{E}_\pi[G_{t+1} \mid S_t=s, A_t=a, R_{t+1}=r, S_{t+1}=s'] \right]
	$$

3. **引入马尔可夫性**（未来仅依赖当前状态 $s'$，与过去的历史无关）：


	$$
	q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \mathbb{E}_\pi[G_{t+1} \mid S_{t+1}=s'] \right]
	$$
   
4. **代回** **$V$** **的定义**（$\mathbb{E}_\pi[G_{t+1} \mid S_{t+1}=s'] = v_\pi(s')$）：


	$$
	q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a) \big[ r + \gamma v_\pi(s') \big]
	$$

---

## 2.3. 贝尔曼期望方程 (Bellman Expectation Equation)

贝尔曼方程的核心是**递归**，即当前价值等于即时奖励的期望加上下一时刻价值的折扣期望。

### 2.3.1.  状态价值的贝尔曼方程及其全展开

$$
v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s]
$$

**全展开形式（将** **$V \to Q \to V$** **结合）：**

$$
v_\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r \mid s, a) \big[ r + \gamma v_\pi(s') \big]
$$

### 2.3.2. 动作价值的贝尔曼方程及其全展开

$$
q_\pi(s, a) = \mathbb{E}_\pi[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a]
$$

**全展开形式（将** **$Q \to V \to Q$** **结合）：**

$$
q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \sum_{a'} \pi(a'|s') q_\pi(s', a') \right]
$$

---

## 2.4. 核心洞察：剥离“双重期望”的单步采样

$v_\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r \mid s, a) \big[ r + \gamma v_\pi(s') \big]$

$q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a) \big[ r + \gamma v_\pi(s') \big]$

其中：

1. **策略的期望**：$\sum_a \pi(a|s)$ （智能体抛骰子选动作）
2. **环境的期望**：$\sum_{s',r} p(s',r|s,a)$ （环境抛骰子给反馈）

在实际交互中，Agent 在状态 $s_t$ 下采样一个动作 $a_t$，环境反馈 $r_{t+1}$ 和 $s_{t+1}$。此时，产生的值 **$r_{t+1} + \gamma v(s_{t+1})$** **的物理意义是：**

1. **$q_\pi(s_t, a_t)$** **剥离了“环境概率”的一次采样**
2. **$v_\pi(s_t)$** **剥离了“策略动作概率”和”环境概率”的一次采样**

💡由于它只是一次采样，不能直接与期望值划等号（即 $v(s_t) \neq r_{t+1} + \gamma v(s_{t+1})$或$q(s_t,a_t) \neq r_{t+1} + \gamma v(s_{t+1})$）。要让采样逼近期望，必须引入**大数定律**与 **Robbins-Monro (RM) 随机近似算法**。

---

## 2.5. **求解Bellman equation**：

- **解析解：**

$$
v_\pi=(I-\gamma P_\pi)^{-1}r_\pi.
$$

- **数值解：**

$$
v_{k+1}=r_\pi+\gamma P_\pi v_k,\quad k=0,1,2,...
$$

$$
v_k\to v_\pi=(I-\gamma P_\pi)^{-1}r_\pi,\quad\text{随着}k\to\infty.
$$

‍

‍
