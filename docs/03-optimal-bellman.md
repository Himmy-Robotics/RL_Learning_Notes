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
