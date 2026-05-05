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
