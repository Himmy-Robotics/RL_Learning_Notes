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
