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
