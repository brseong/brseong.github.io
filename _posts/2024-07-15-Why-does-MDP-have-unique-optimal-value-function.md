---
layout: post
title:  Why does MDP have unique optimal value function?
categories: [Machine Learning, Reinforcement Learning]
use_math: true
---

Many lectures or articles discussing reinforcement learning claim that MDPs have a unique optimal value function. But why does it?

First, we need to define what policy is optimal. [1, 3]

**Definition 1**. $\pi_*$ is an optimal policy if $\forall\pi:\forall s\in\mathcal{S}:v_{\pi_*}(s)\ge v_\pi(s).$

We can derive the fact below by definition of optimal policy. [4]
$$\forall \pi_1,\pi_2:\forall s\in\mathcal{S}:v_{\pi_1}(s)=v_{\pi_2}(s).$$

But it may be very confusing. Why can't multiple optimal value functions exist? To understand this, we have to start from the definition of contraction mapping. The definition below is from Wikipedia. [6]

**Definition 2**. In mathematics, a contraction mapping, or contraction or contractor, on a metric space ($M$, $d$) is a function $f$ from $M$ to itself, with the property that there is some real number $0\le k<1$ such that for all $x$ and $y$ in $M$,
$$d(f(x),f(y))\le kd(x,y).$$
Now, we will treat value functions as vectors. (In continuous state space, imagine an infinite dimension vector.) Then, we can define the Bellman optimality operator $T$. [1]

**Definition 3**. Bellman optimality operator is a vector-to-vector function, satisfying the equation below:
$$T[\boldsymbol{v}](s)=\max_{a\in\mathcal{A}(s)}\mathbb{E}\left[R_{t+1}+\gamma v(S_{t+1})\middle|S_t=s,A_t=a\right]$$
Bellman optimality operator means a greedy value function improvement, a step of value iteration.
Now, we can show that $T$ is a contraction mapping in the max norm. We need the definition of the max norm and additional lemma. [1]

**Definition 4**. $\|\boldsymbol{v}-\boldsymbol{v}'\|_\infty=\max_{s\in\mathcal{S}}|v(s)-v'(s)|.$

**Lemma 5**. Given a finite set $\mathcal{A}$, for any two real value functions $f_1:\mathcal{A}\to\mathbb{R}$ and $f_2:\mathcal{A}\to\mathbb{R}$ we have
$$|\max_{a\in\mathcal{A}}f_1(a)-\max_{a\in\mathcal{A}}f_2(a)|\le\max_{a\in\mathcal{A}}|f_1(a)-f_2(a)|.$$

Now, we can derive the target theorem.

**Theorem 6**. On the metric space with max-norm, $T$ is $\gamma$ contraction.

*Proof*.
$$\begin{align} \left\|T[\boldsymbol{v}]-T[\boldsymbol{v}']\right\|_\infty&=\max_{s\in\mathcal{S}}|T[\boldsymbol{v}](s)-T[\boldsymbol{v}'](s)|\\ &=\max_{s\in\mathcal{S}}\left|\max_{a\in\mathcal{A}(s)}\mathbb{E}\left[R_{t+1}+\gamma v(S_{t+1})\middle|s,a\right]-\max_{a\in\mathcal{A}(s)}\mathbb{E}\left[R_{t+1}+\gamma v'(S_{t+1})\middle|s,a\right]\right|\because\text{Definition 3}\\ &\le\max_{s\in\mathcal{S}}\max_{a\in\mathcal{A}(s)}\left|\mathbb{E}\left[R_{t+1}+\gamma v(S_{t+1})\middle|s,a\right]-\mathbb{E}\left[R_{t+1}+\gamma v'(S_{t+1})\middle|s,a\right]\right|\because\text{Lemma 5}\\ &=\max_{s\in\mathcal{S}}\max_{a\in\mathcal{A}(s)}\left|\gamma\mathbb{E}\left[v(S_{t+1}) - v'(S_{t+1})\middle|s,a\right]\right|\\ &\le\max_{s\in\mathcal{S}}\max_{a\in\mathcal{A}(s)}\gamma\mathbb{E}\left[\left|v(S_{t+1}) - v'(S_{t+1})\right|\,\middle|s,a\right]\because\text{Jensen inequality}\\ &\le\max_{s\in\mathcal{S}}\max_{a\in\mathcal{A}(s)}\gamma\mathbb{E}\left[\max_{s'\in\mathcal{S}}\left|v(s') - v'(s')\right|\,\middle|s,a\right]\\ &=\max_{s\in\mathcal{S}}\max_{a\in\mathcal{A}(s)}\gamma\mathbb{E}\left[\left\|\boldsymbol{v} - \boldsymbol{v}'\right\|_\infty\middle|s,a\right]\because\text{Definition 4}\\ &=\gamma\left\|\boldsymbol{v} - \boldsymbol{v}'\right\|_\infty\because\text{Expectation is constant} \end{align}$$
It is guaranteed to converge into a unique fixed point by Banach fixed-point theorem. (Also known as contraction mapping theorem.) [1, 3, 7] In other words, a unique value function exists that satisfies $T[\boldsymbol{v}]=\boldsymbol{v}$. Considering that $T$ means a greedy value function improvement, it can interpreted to the conclusion that:
There is only one optimal valeu function, and we can get the optimal value function by applying the value iteration to any value function.

References and related literatures
1. https://towardsdatascience.com/why-does-the-optimal-policy-exist-29f30fd51f8c
2. Silver, D., Reinforcement Learning, Lecture Slides, https://www.davidsilver.uk/teaching/, 2015
3. lecture5-2pp.pdf (berkeley.edu)
4. OptimalPolicyExistence.pdf (stanford.edu)
5. reinforcement learning - Uniqueness of the optimal value function for an MDP - Cross Validated (stackexchange.com)
6. Contraction mapping - Wikipedia
7. Banach fixed-point theorem - Wikipedia