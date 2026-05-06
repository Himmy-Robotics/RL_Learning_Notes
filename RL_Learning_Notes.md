# 强化学习的数学原理

# 1. Basic Concepts

<u>本章是强化学习的一些基本先验知识，不做过多介绍。</u>

‍

**Reward：** $r(s, a)$ 是状态s和a的函数

**Return：**  一条trajectory的奖励之和

**Discount return：**  对于一条trajectory而言 $discount return = r_{1} + \gamma r_{2} + \gamma^{2} r_{3} + ...$

‍

**MDP—Markov decision processes**

**M：**  Markov property(memoryless property)

$$
p(s_{t+1}|s_t,a_t,s_{t-1},a_{t-1},\ldots,s_0,a_0)=p(s_{t+1}|s_t,a_t),
p(r_{t+1}|s_t,a_t,s_{t-1},a_{t-1},\ldots,s_0,a_0)=p(r_{t+1}|s_t,a_t),
$$

**D:**   Stochastic Policy

‍

‍

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

# 3. Optimal State Values and Bellman Optimality Equation

<u>本章介绍关于强化学习中最优Value以及贝尔曼最优公式。</u>

‍

**Optimal policy:**   考虑策略$\pi^*$,如果对任意的状态$s\in\mathcal{S}$和其他任意策略 $\pi$,都有 $v_\pi^*(s)\geqslant v_\pi(s)$,那么 $\pi^*$是一个最优策略，并且$\pi^*$对应的状态值是最优状态值。

**Bellman optimality equation:**

$$
\begin{aligned}
v(s) & =\max_{\pi(s)\in\Pi(s)}\sum_{a\in\mathcal{A}}\pi(a|s)\left(\sum_{r\in\mathcal{R}}p(r|s,a)r+\gamma\sum_{s^{\prime}\in\mathcal{S}}p(s^{\prime}|s,a)v(s^{\prime})\right) \\
 & =\max_{\pi(s)\in\Pi(s)}\sum_{a\in\mathcal{A}}\pi(a|s)q(s,a),
\end{aligned}
$$

**Bellman optimality equation的解:**

始终存在唯一解$v^*$,该解可以通过如下迭代算法求解：

$$
v_{k+1}=f(v_k)=\max_{\pi\in II}(r_\pi+\gamma P_\pi v_k),\quad k=0,1,2,\ldots
$$

对任意给定的$v_0$,当$k\to\infty$时，$v_k$和$\pi_{k}$以指数收敛到最优的状态值和策略$v^*,\pi^*$。

💡**由于贝尔曼最优公式满足压缩映射定理，因此其解总是存在，同时该解存在唯一性。**

**Optimal policy的解：**

$$
\pi^*=\arg\max_{\pi\in\Pi}(r_\pi+\gamma P_\pi v^*).
$$

‍

‍

# 4. Value Iteration and Policy Iteration

<u>本节的算法为model based的算法，这些算法也被成为动态规划。</u>

‍

## 4.1. **Value iteration:**

**值迭代的公式：**

$$
v_{k+1}=\max_{\pi\in II}(r_\pi+\gamma P_\pi v_k),\quad k=0,1,2,\ldots
$$

对任意给定的$v_0$,当$k\to\infty$时，$v_k$和$\pi_{k}$以指数收敛到最优的状态值和策略$v^*,\pi^*$。

1. **策略更新**

$$
\pi_k=\arg\max_{\pi\in\Pi}(r_\pi+\gamma P_\pi v_k).
$$

其中$v_k$是上次迭代得到的值。

2. **值迭代**

$$
v_{k+1}=r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_k,
$$

 **💡值迭代过程中求出来的并不是state value。**

---

## 4.2. **Policy** **iteration:**

1. **策略评价：**

$$
v_{\pi_k}=r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k},
$$

在计算$v_{\pi_k}$是需要用到迭代的方法。

$$
v_{\pi_k}^{(j+1)}=r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k}^{(j)},\quad j=0,1,2,....
$$

直到$j\to\infty$，所求出来的就是state value。

2. **策略改进：**

$$
\pi_{k+1}=\arg\max_\pi(r_\pi+\gamma P_\pi v_{\pi_k}).
$$

使用第一步求出的$v_{\pi_k}$来更新策略。

---

## 4.3. **Truncated policy iteration：**

对于Policy iteration而言，其中的策略评价阶段只需要$j=j_{\mathrm{truncate}}$即可。

‍

‍

# 5. Monte Carlo Methods

<u>从本章起，开始介绍Model free的强化学习方法。</u>​**<u>没有模型 → 必须有数据（经验样本）。</u>**​<u>  用样本均值近似期望值来替代模型计算。</u>

## 5.1. MC估计的定义：

对于随机变量X，估计其期望值$\mathbb{E}[X]$:

- **有模型时直接根据期望的定义给出：**

$$
\mathbb{E}[X] = \sum_{x \in \mathcal{X}} p(x)x.
$$

- **无模型时根据根据MC估计（大数定理保证n趋于无穷时估计的准确性）：**

$$
\mathbb{E}[X] \approx \bar{x} = \frac{1}{n} \sum_{j=1}^{n} x_j.
$$

---

## 5.2. MC方法在强化学习中的应用：

### 5.2.1. **MC basic:**

将策略迭代中有模型的策略评估替换为无模型的MC估计,直接估算action value。

$$
q_{\pi_k}(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a] \approx \frac{1}{n} \sum_{i=1}^{n} g_{\pi_k}^{(i)}(s, a)
$$

对每个 $(s, a)$，收集足够多的episode，用回报均值近似 $q_{\pi_k}(s, a)$。

### 5.2.2. **MC Exploring Starts:**

- 增加样本利用效率：使用Every-visit，每个 $(s, a)$ **每次**出现时都用后续轨迹估计（样本效率最高，但样本有相关性）。

- 策略更新频率：不等所有episode收集完，每收到一个episode就立即更新（广义策略迭代思想）。

### 5.2.3. **MC ε-Greedy:**

将policy take action的过程从**Greedy**改为**ε-Greedy**，软策略保证每个$(s, a)$都有正概率被访问 → 足够长的单条episode即可覆盖所有状态-动作对。

**ε-Greedy policy:**

$$
\pi(a|s) = \begin{cases} 1 - \frac{\epsilon}{|\mathcal{A}(s)|}(|\mathcal{A}(s)| - 1), & a = a^* \\ \frac{\epsilon}{|\mathcal{A}(s)|}, & a \neq a^* \end{cases}
$$

**收敛性**：收敛到ε-greedy策略集合中的最优策略，但未必是全局最优。ε足够小时接近全局最优。

‍

‍

# 6. Stochastic Approximation

<u>本章介绍的是强化学习中使用采样到的样本来估计随机变量分布的随机近似方法，这种对于随机变量的近似估计方法贯穿了整个强化学习过程。</u>

## 6.1. 随机近似方法分类

从宏观角度上来讲随机近似可以分为非增量式和增量式两种，非增量式也就是MC方法，一种是增量式的方法，TD方法就是一种典型的增量式的方法。

- **非增量式**（MC方法，收集全部样本后计算）：

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

- **增量式**（每收到一个样本即更新）：

$$
w_{k+1} = w_k - \frac{1}{k}(w_k - x_k)
$$

可验证$w_{k+1} = \frac{1}{k}\sum_{i=1}^{k} x_i$，与非增量式等价。

- **一般形式：**

$$
w_{k+1} = w_k - \alpha_k(w_k - x_k)
$$

---

## 6.2. **Robbins-Monro (RM) 算法：**

**要解决的问题：**

想象你面前有一个黑箱子。你往里面输入一个数 $w$，箱子会吐出一个数，但这个数不是精确的 $g(w)$，而是被噪声污染过  
的$\tilde g(w, \eta) = g(w) + \eta_0$。你不知道 $g$ 长什么样，也不知道它的导数，你唯一能做的就是：输入一个 $w$，观察一个带噪声的输出。你的目标是：找到那个让 $g(w) = 0$ 的 $w^*$。

**本质：** ​  *“新的猜测*    *=*    *旧的猜测 - 步长 × 当前观测到的输出”*

**一般形式：**

$$
w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)
$$

**用于均值估计：**  比如我们想要求随机变量X的期望，令 $g(w) = w - \mathbb{E}[X]=0$，$w^* = \mathbb{E}[X]$,观测$\tilde{g} = w_k - x_k$，噪声则为$\eta_k = \mathbb{E}[X] - x_k$则：

$$
w_{k+1} = w_k - \alpha_k(w_k - x_k)
$$

 **💡RM算法的收敛性保证了增量式随机近似方法用于估计随机变量的有效性。**

---

## 6.3. **SGD:**

**求解的优化问题：**

$$
\min_w J(w) = \mathbb{E}[f(w, X)]
$$

**算法与更新公式：**

|算法|更新公式|
| ----------------| ---------------------|
|梯度下降（GD）|$w_{k+1} = w_k - \alpha_k \mathbb{E}[\nabla_w f(w_k, X)]$ （需要知道分布）|
|SGD|$w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)$ （用样本替代期望）|
|MBGD|$w_{k+1} = w_k - \frac{\alpha_k}{m} \sum_{j \in \mathcal{I}_k} \nabla_w f(w_k, x_j)$|
|BGD|$w_{k+1} = w_k - \frac{\alpha_k}{n} \sum_{i=1}^{n} \nabla_w f(w_k, x_i)$|

**SGD与RM的关系：**

**SGD是特殊的RM算法：**

令：

$$
g(w) = \nabla_w J(w) = \mathbb{E}[\nabla_w f(w, X)]
$$

观测值：

$$
\tilde g = \nabla_w f(w, x_k), \quad \text{噪声为 } \eta = \nabla_w f(w, x) - \mathbb{E}[\nabla_w f(w, X)]
$$

则 SGD 就是 RM 算法求 $g(w) = 0$ 的根。

‍

‍

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

# 8. Value Function Approximation

<u>本章的关键转变是</u>​**<u>从表格表示（tabular）</u>**​**<u>$\to$**</u>​<u> </u>​**<u>函数近似表示（function approximation）</u>**​<u> 。之前所有章节中，状态/动作价值都存储在一张表里。当状态空间很大时，表格方法不可行。本章用参数化函数 </u>​<u>$\hat{v}(s, w)$</u>​<u> 来近似价值，其中 </u>​<u>$w$</u>​<u> 是需要学习的参数向量。这也是神经网络进入强化学习的入口。</u>

**与使用表格的方式进行对比：**

- 在使用表格更新价值时，每次改变一个条目，其他的不变，使用函数近似的时候，每次改变的是参数$w$，改变$w$后，所有状态的价值都可能会被改变，这也带来了一定的泛化能力。

- **函数近似不一定能精确表示所有状态价值，存在近似误差。**

‍

## 8.1. TD-learning + Function approximation

**目标函数构建：**

找最优 $w$ 使 $\hat{v}(s, w)$ 尽可能接近真实值 $v_\pi(s)$：

$$
J(w) = \sum_{s \in \mathcal{S}} d_\pi(s) (v_\pi(s) - \hat{v}(s, w))^2
$$

其中 $d_\pi(s)$ 是策略 $\pi$ 下的**平稳分布**（stationary distribution），表示 agent 长期运行后访问各状态的概率。访问频率高的状态被赋予更大权重。

**目标函数优化：**

对目标函数$J(w) = \mathbb{E} \left[ (v_\pi(S) - \hat{v}(S, w))^2 \right]$做梯度下降,化简后得到：

- *TD+函数近似（一般形式）：*

$$
w_{t+1} = w_t + \alpha_t \left[ r_{t+1} + \gamma \hat{v}(s_{t+1}, w_t) - \hat{v}(s_t, w_t) \right] \nabla_w \hat{v}(s_t, w_t)
$$

- *TD-Linear（线性情况）：*  *$\hat{v}(s, w) = \phi^T(s)w$*  *，*  *$\nabla_w \hat{v} = \phi(s)$*  *，代入得*  *：*

$$
w_{t+1} = w_t + \alpha_t \left[ r_{t+1} + \gamma \phi^T(s_{t+1})w_t - \phi^T(s_t)w_t \right] \phi(s_t)
$$

- *TD-Linear 算法的确定性等价形式简化为：*

$$
w_{t+1} = w_t + \alpha_t(b - Aw_t)
$$

<span data-type="text" style="font-size: 15px;">其中 </span>$A = \Phi^T D(I - \gamma P_\pi)\Phi$​<span data-type="text" style="font-size: 15px;">，</span>$b = \Phi^T D r_\pi$​<span data-type="text" style="font-size: 15px;">。收敛到 </span>$w^* = A^{-1}b$​<span data-type="text" style="font-size: 15px;">。</span>

**LSTD(最小二乘TD)：**

直接用样本估计 $A$ 和 $b$，然后计算 $w^* \approx \hat{A}^{-1}\hat{b}$：

$$
\hat{A}_t = \sum_{k=0}^{t-1} \phi(s_k)(\phi(s_k) - \gamma\phi(s_{k+1}))^T, \quad \hat{b}_t = \sum_{k=0}^{t-1} r_{k+1}\phi(s_k)
$$

**优点：** 样本利用效率高，收敛快。缺点：仅限线性情况，计算复杂度较高（需要矩阵求逆）。

---

## 8.2. Action value estimation + Function approximation

### 8.2.1. **Sarsa + Function approximation:**

用 $\hat{q}(s, a, w) = \phi^T(s, a)w$ 近似动作价值，将状态价值的 TD 公式中替换为动作价值：

$$
w_{t+1} = w_t + \alpha_t \left[ r_{t+1} + \gamma \hat{q}(s_{t+1}, a_{t+1}, w_t) - \hat{q}(s_t, a_t, w_t) \right] \nabla_w \hat{q}(s_t, a_t, w_t)
$$

结合 $\epsilon\text{-greedy}$ 策略改进，可以学习最优策略。

### 8.2.2. **Q-learning + Function approximation:**

$$
w_{t+1} = w_t + \alpha_t \left[ r_{t+1} + \gamma \max_{a} \hat{q}(s_{t+1}, a, w_t) - \hat{q}(s_t, a_t, w_t) \right] \nabla_w \hat{q}(s_t, a_t, w_t)
$$

与 Sarsa 的唯一区别：TD 目标中用 $\max_a$ 替代 $\hat{q}(s_{t+1}, a_{t+1})$。

💡虽然价值用函数表示了，但此时**策略仍然用表格表示**，因此仍假设有限状态和动作空间。第 9 章将引入策略函数来处理连续空间

---

## 8.3. Deep Q-Learning

**目标函数：**

Q-learning是要求最优的Q值，从而找到最优policy，最优的Q值通过贝尔曼最优公式可以表示为：

$$
q^*(s, a) = \mathbb{E} [R_{t+1} + \gamma \max_{a'} q^*(S_{t+1}, a') \mid S_t = s, A_t = a]
$$

因此将目标函数定义为最优的Q值与用于拟合的函数的差值的平方：

$$
J = \mathbb{E} \left[ \left( R + \gamma \max_{a \in \mathcal{A}(S')} \hat{q}(S', a, w) - \hat{q}(S, A, w) \right)^2 \right]
$$

**技巧一：双网络 (Main Network + Target Network)：**

$w$ 同时出现在 $\hat{q}(S, A, w)$ 和 $\max_a \hat{q}(S', a, w)$ 中，梯度难以计算。解决方法：引入目标网络 $\hat{q}(S, a, w_T)$，其参数 $w_T$ 在一段时间内固定不变。梯度简化为：

$$
\nabla_w J = -\mathbb{E} \left[ \left( R + \gamma \max_a \hat{q}(S', a, w_T) - \hat{q}(S, A, w) \right) \nabla_w \hat{q}(S, A, w) \right]
$$

- **主网络** **$w$**：每步都更新。
- **目标网络** **$w_T$**：每隔 $C$ 步同步为 $w_T = w$。

**技巧二：经验回放 (Experience Replay)：**

- 将经验样本 $(s, a, r, s')$ 存入回放缓冲区 $\mathcal{B}$。
- 每次更新时从 $\mathcal{B}$ 中**均匀随机**抽取一个小批量。
- **为什么需要**：目标函数假设 $(S, A)$ 均匀分布，但实际样本是按行为策略序列生成的（有相关性）。均匀随机抽样打破了样本间的相关性。
- **额外好处**：每个样本可以被多次使用，提高数据效率。

**训练流程(off-policy版本)：**

1. 用行为策略 $\pi_b$ 生成经验样本，存入回放缓冲区。
2. 每次迭代从缓冲区均匀抽取小批量样本。
3. 对每个样本计算目标值 $y_T = r + \gamma \max_a \hat{q}(s', a, w_T)$。
4. 训练主网络最小化 $\sum (y_T - \hat{q}(s, a, w))^2$。
5. 每隔 $C$ 步令 $w_T = w$。

---

‍

‍

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

# 11. PPO（Proximal Policy Optimization）

<u>本章介绍PPO，PPO要解决的问题说到底就是使用了重要性采样使得Policy Gradient由on-policy变为off-policy的时候，用于sample的policy的参数和agent的policy的参数不能相差太大，因此通过引入KL散度，为了让两个policy的参数尽可能接近，通过引入KL散度的方法的不同又可以分为TRPO，PPO-Penalty，PPO-Clip。</u>

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
