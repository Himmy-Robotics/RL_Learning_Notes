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
