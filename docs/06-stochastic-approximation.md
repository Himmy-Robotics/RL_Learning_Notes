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
