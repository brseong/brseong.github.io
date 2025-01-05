---
layout: post
title: Pattern Recognition and Machine Learning Study (Ch9).
categories: [Machine Learning, Probability]
use_math: true
---

[STDP enables spiking neurons to detect hidden causes of their inputs](https://proceedings.neurips.cc/paper/2009/hash/a5cdd4aa0048b187f7182f1b9ce7a6a7-Abstract.html){: target="_blank"}를 공부하고 있는데, 3번째 문단인 Underlying theoretical principles에 Expectation Maximization (EM) 알고리즘이 등장하여 공부할 필요성을 느꼈다. Bishop의 패턴 인식 (PRML)의 EM 파트는 간략히는 흝어봤었으나, 그 이론적 기반을 제대로 이해하지는 못했기 때문에, 다시 제대로 공부하기로 하였다. 이 글에는 9 챕터를 읽으면서 남은 의문들이나 내가 이해한 바를 서술하였다.

# 9.2.1
K-means 클러스터링은 따로 어려울 게 없었으나, 9.2.1 챕터부터 이해하기 어려운 것들이 등장하였다. 특히 일부 서술의 경우에는 머리를 싸맸으나 별다른 답을 찾지 못하였다. ([다변량 가우시안의 붕괴에 대한 서술이라든지.]({{site.baseurl}}/questions))

# 9.2.2
이 챕터에서는 다변수 미적분을 이용해 노가다하면 식 9.16인

$$0=\sum_{n=1}^N\frac{\pi_k\mathcal{N}(\underline{x}_n|\underline{\mu}_k,\underline{\underline{\Sigma}}_k)}{\sum_j\pi_j\mathcal{N}(\underline{x}_n|\underline{\mu}_j,\underline{\underline{\Sigma}}_j)}\underline{\underline{\Sigma}}^{-1}_k(\underline{x}_n-\underline{\mu}_k)$$

을 쉽게 유도할 수 있다. (텐서 랭크를 언더바로 표시해 책과 표기가 다를 수 있다.) 식 9.17, 9.18도 쉽게 유도가 된다.

하지만 식 9.19에서 막혔는데, 이를 유도하기 위해서는 

$$\partial_{\underline{\underline{\Sigma}}_k}\ln p(\underline{\underline{x}}|\underline{\underline{\mu}},\underline{\underline{\underline{\Sigma}}}) = \partial_{\underline{\underline{\Sigma}}_k} \sum_{n=1}^N\ln\sum_{k'=1}^{K}\pi_{k'}\frac{1}{\det(\underline{\underline{\Sigma}})^{\frac12}(2\pi)^{\frac{d}{2}}}\exp{-\frac{1}{2}}\Sigma^{-1}_{k'pq}(x_{np}-\mu_{k'p})(x_{nq}-\mu_{k'q})$$

를 계산해야 한다. 그런데 이를 계산하기 위해서는 몇 가지 준비사항이 필요하다. 아래 행렬 미분 식은 PRML appendix C에 실린 공식들의 유도이다.

- 역행렬의 미분 (C.21)
  
    $$\frac{\partial I_{jk}}{\partial x}=\frac{\partial [AA^{-1}]}{\partial x}=\frac{\partial A}{\partial x}A^{-1}+A\frac{\partial A^{-1}}{\partial x} = 0$$
    $$\implies A\frac{\partial A^{-1}}{\partial x}=-\frac{\partial A}{\partial x}A^{-1}$$
    $$\implies \frac{\partial A^{-1}}{\partial x}=-A^{-1}\frac{\partial A}{\partial x}A^{-1}$$

- $\frac{\partial}{\partial A}\text{Tr}(AB)=B^\top$ (C.23)
    $$\frac{\partial A_{pr}B_{rp}}{\partial A_{ab}}=\frac{\partial A_{pr}}{\partial A_{ab}}B_{rp}+A_{pr}\frac{\partial B_{rp}}{\partial A_{ab}}$$
    $$=\delta_{ap}\delta_{br}B_{rp}+0_{ab}$$
    $$=B_{ba}=B^\top$$

제일 어려웠던 유도인 C.22이다. 교재에는 답지가 나와있지 않아서, 많이 고생했다.
- $\frac{\partial}{\partial x}\ln\det(A)=\text{Tr}(A^{-1}\frac{\partial A}{\partial x})$ (C.22)
  $$\frac{\partial}{\partial x}\ln|A|=\frac{\partial}{\partial x}\sum_{n=1}^N\ln\lambda_n$$
  $$\begin{align}
    =\sum_{n=1}^N\frac{1}{\lambda_n}\frac{\partial \lambda_n}{\partial x}
  \end{align}$$

    Assuming $A=A^\top$, then $A$ is orthogonally diagonalizable: $A=U\Lambda U^\top=\sum_{n=1}^N\lambda_n\mathbf{u}_n\mathbf{u}_n^\top.$ Differentiate this equation by $x$.
    $$\frac{\partial A}{\partial x} = \sum_{n=1}^N\left[\frac{\partial\lambda_n}{\partial x}\mathbf{u}_n\mathbf{u}_n^\top+\lambda_n\frac{\partial\mathbf{u}_n}{\partial x}\mathbf{u}_n^\top+\lambda_n\mathbf{u}_n\frac{\partial\mathbf{u}_n^\top}{\partial x}\right]$$
    $$\begin{align}
        \implies A^{-1}\frac{\partial A}{\partial x}&=\sum_{n=1}^N\left[\frac{\partial\lambda_n}{\partial x}A^{-1}\mathbf{u}_n\mathbf{u}_n^\top+\lambda_nA^{-1}\frac{\partial\mathbf{u}_n}{\partial x}\mathbf{u}_n^\top+\lambda_nA^{-1}\mathbf{u}_n\frac{\partial\mathbf{u}_n^\top}{\partial x}\right]\nonumber\\
        &=\sum_{n=1}^N\left[\frac{1}{\lambda_n}\frac{\partial\lambda_n}{\partial x}\mathbf{u}_n\mathbf{u}_n^\top+\lambda_nA^{-1}\frac{\partial\mathbf{u}_n}{\partial x}\mathbf{u}_n^\top+\lambda_n\frac{1}{\lambda_n}\mathbf{u}_n\frac{\partial\mathbf{u}_n^\top}{\partial x}\right]
    \end{align}$$

    Since $U$ is orthogonal, $\forall_n \mathbf{u}_n^\top\mathbf{u}_n=1.$ Then,
    $$
        \frac{\partial [\mathbf{u}_n^\top\mathbf{u}_n]}{\partial x}=
            \frac{\partial \mathbf{u}_n^\top}{\partial x}\mathbf{u}_n
            +
            \mathbf{u}_n^\top\frac{\partial \mathbf{u}_n}{\partial x}=0.$$

    Because $\left(\frac{\partial \mathbf{u}_n^\top}{\partial x}\mathbf{u}_n\right)^\top=\mathbf{u}_n^\top\frac{\partial \mathbf{u}_n}{\partial x}$ and $\mathbf{u}_n^\top\frac{\partial \mathbf{u}_n}{\partial x}\in\mathbb{R},$
    $$\begin{align}
            \mathbf{u}_n^\top\frac{\partial \mathbf{u}_n}{\partial x}=0.
    \end{align}$$
    <!-- $$\implies\text{Tr}\left(\frac{\partial A}{\partial x}\right)=\left(\frac{\partial A}{\partial x}\right)_{ii}=\frac{\partial\lambda_i}{\partial x}\mathbf{u}_{ij}\mathbf{u}_{ji}=\sum_{n=1}^N\frac{\partial\lambda_n}{\partial x} \cdots (2)$$ -->
    <!-- $$\because \mathbf{u}_{ij}\mathbf{u}_{ji}=I \text{ since orthogonal eigenvectors.}$$ -->
    
    Continued from $(2)$,
    $$\begin{align}
        \text{Tr}\left(A^{-1}\frac{\partial A}{\partial x}\right)=&\sum_{p=1}^N\left(A^{-1}\frac{\partial A}{\partial x}\right)_{pp}
        \nonumber\\
        =&\sum_{p=1}^P\left(\sum_{n=1}^N\left[\frac{1}{\lambda_n}\frac{\partial\lambda_n}{\partial x}\mathbf{u}_n\mathbf{u}_n^\top+\lambda_nA^{-1}\frac{\partial\mathbf{u}_n}{\partial x}\mathbf{u}_n^\top+\mathbf{u}_n\frac{\partial\mathbf{u}_n^\top}{\partial x}\right]\right)_{pp}\nonumber\\
        =&\sum_{p=1}^P\sum_{n=1}^N\left[\frac{1}{\lambda_n}\frac{\partial\lambda_n}{\partial x}\mathbf{u}_n\mathbf{u}_n^\top+\lambda_nA^{-1}\frac{\partial\mathbf{u}_n}{\partial x}\mathbf{u}_n^\top+\mathbf{u}_n\frac{\partial\mathbf{u}_n^\top}{\partial x}\right]_{pp}\nonumber\\
        =&\sum_{n=1}^N\sum_{p=1}^P\left[\frac{1}{\lambda_n}\frac{\partial\lambda_n}{\partial x}\mathbf{u}_n\mathbf{u}_n^\top+\lambda_nA^{-1}\frac{\partial\mathbf{u}_n}{\partial x}\mathbf{u}_n^\top+\mathbf{u}_n\frac{\partial\mathbf{u}_n^\top}{\partial x}\right]_{pp}\nonumber\\
        =&\sum_{n=1}^N\sum_{p=1}^P\left[
            \frac{1}{\lambda_n}\frac{\partial\lambda_n}{\partial x}u_{np}u_{np}
            +
            \lambda_nA^{-1}_{pk}\frac{\partial u_{nk}}{\partial x}u_{np}
            +
            u_{np}\frac{\partial u_{np}}{\partial x}
            \right]\nonumber\\
        =&\sum_{n=1}^N\left[
            \frac{1}{\lambda_n}\frac{\partial\lambda_n}{\partial x}
            +
            \sum_{p=1}^P\left[
                \lambda_nu_{np}(A^{-1})^\top_{pk}\frac{\partial u_{nk}}{\partial x}
                +
                u_{np}\frac{\partial u_{np}}{\partial x}
                \right]
            \right]
            \qquad \because\ A^\top=A\implies (A^{-1})^\top=(A^\top)^{-1}\nonumber\\
        =&\sum_{n=1}^N\left[
            \frac{1}{\lambda_n}\frac{\partial\lambda_n}{\partial x}
            +
            \lambda_n\mathbf{u}_{n}^\top(A^{-1})^\top\frac{\partial \mathbf{u}_{n}}{\partial x}
            +
            \mathbf{u}_{n}^\top\frac{\partial \mathbf{u}_{n}}{\partial x}
            \right]\nonumber\\
        =&\sum_{n=1}^N\left[
            \frac{1}{\lambda_n}\frac{\partial\lambda_n}{\partial x}
            +
            \lambda_n(A^{-1}\mathbf{u}_n)^\top\frac{\partial \mathbf{u}_{n}}{\partial x}
            \right]
            \qquad\because\ (3)\nonumber\\
        =&\sum_{n=1}^N\left[
            \frac{1}{\lambda_n}\frac{\partial\lambda_n}{\partial x}
            +
            \lambda_n(\frac{1}{\lambda_n}\mathbf{u}_n)^\top\frac{\partial \mathbf{u}_{n}}{\partial x}
            \right]\nonumber\\
        =&\sum_{n=1}^N\left[
            \frac{1}{\lambda_n}\frac{\partial\lambda_n}{\partial x}
            +
            \mathbf{u}_n^\top\frac{\partial \mathbf{u}_{n}}{\partial x}
            \right]\nonumber\\
        =&\sum_{n=1}^N\left[
            \frac{1}{\lambda_n}\frac{\partial\lambda_n}{\partial x}
            \right]\qquad\because\ (3)\nonumber\\
        =&(1)\nonumber\\
        =&\frac{\partial}{\partial x}\ln\det(A). \qquad\Box\nonumber
    \end{align}$$

- $\frac{\partial}{\partial A}\ln \det(A)=A^{-\top}$
  $$\begin{align}
    \frac{\partial}{\partial A_{pq}}\ln\det(A)=&
        \text{Tr}\left(A^{-1}\frac{\partial A}{\partial A_{pq}}\right)\nonumber\\
        =&\left(A^{-1}\frac{\partial A}{\partial A_{pq}}\right)_{ii}\nonumber\\
        =&A^{-1}_{ij}\frac{\partial A_{ji}}{\partial A_{pq}}\nonumber\\
        =&A^{-1}_{ij}\delta_{pj}\delta_{qi}\nonumber\\
        =&A^{-1}_{qp}\nonumber\\
        =&A^{-\top}_{pq}\qquad\Box\nonumber
  \end{align}$$

  *계속 작성 중.*