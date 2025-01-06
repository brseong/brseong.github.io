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

을 쉽게 유도할 수 있다. (텐서 랭크를 언더바로 표시해 책과 표기가 다를 수 있다. 그닥 엄밀하지는 않지만 밑줄이 하나면 벡터, 밑줄이 두개면 행렬 이런 식으로 생각하면 된다.) 식 9.17, 9.18도 쉽게 유도가 된다.

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

# 2.3.4
식 9.19 옆에 참조로 2.3.4 챕터가 있어 찾아보니, 식 2.122가 굉장히 유사한 모양을 지니고 있었다. 따라서 식 9.19를 먼저 유도해볼 필요성을 느꼈다.

$$\begin{align*}
    \frac{\partial}{\partial \Sigma_{pq}}\ln p(\mathbf{X}|\mathbf{\mu},\mathbf{\Sigma})
    =&\frac{\partial}{\partial \Sigma_{pq}}\sum_{n=1}^N\left[
        -\frac{1}{2}\ln|\mathbf{\Sigma}|
        -
        \frac{N}{2}\ln2\pi
        -
        \frac{1}{2}(\mathbf{\mathbf{x}_n-\mathbf{\mu}})^\top\mathbf{\Sigma}^{-1}(\mathbf{\mathbf{x}_n-\mathbf{\mu}})
        \right]\\
    =&\sum_{n=1}^N\left[
        -\frac{1}{2}\frac{\partial}{\partial \Sigma_{pq}}\left[\ln|\mathbf{\Sigma}|\right]
        -
        \frac{1}{2}(\mathbf{\mathbf{x}_n-\mathbf{\mu}})^\top\frac{\partial}{\partial \Sigma_{pq}}\left[\mathbf{\Sigma}^{-1}\right](\mathbf{\mathbf{x}_n-\mathbf{\mu}})
        \right]\\
    =&\sum_{n=1}^N\left[
        -\frac{1}{2}\text{Tr}\left(\mathbf{\Sigma}^{-1}\frac{\partial\mathbf{\Sigma}}{\partial\Sigma_{pq}}\right)
        -
        \frac{1}{2}(\mathbf{\mathbf{x}_n-\mathbf{\mu}})^\top\mathbf{\Sigma}^{-1}\frac{\partial \mathbf{\Sigma}}{\partial \Sigma_{pq}}\mathbf{\Sigma}^{-1}(\mathbf{\mathbf{x}_n-\mathbf{\mu}})
        \right]\qquad\because\ \text{C.21, C.22}\\
    =&\sum_{n=1}^N\left[
        -\frac{1}{2}\left(\mathbf{\Sigma}^{-1}\frac{\partial\mathbf{\Sigma}}{\partial\Sigma_{pq}}\right)_{ii}
        -
        \frac{1}{2}(\mathbf{x}_{ni}-\mu_{i})\Sigma^{-1}_{ij}\frac{\partial \Sigma_{jk}}{\partial \Sigma_{pq}}\Sigma^{-1}_{kl}(x_{nl}-\mu_{l})
        \right]\\
    =&\sum_{n=1}^N\left[
        -\frac{1}{2}\Sigma^{-1}_{ij}\frac{\partial\Sigma_{ji}}{\partial\Sigma_{pq}}
        -
        \frac{1}{2}(\mathbf{x}_{ni}-\mu_{i})\Sigma^{-1}_{ij}\delta_{pj}\delta_{qk}\Sigma^{-1}_{kl}(x_{nl}-\mu_{l})
        \right]\\
    =&\sum_{n=1}^N\left[
        -\frac{1}{2}\Sigma^{-1}_{ij}\delta_{pj}\delta_{qi}
        -
        \frac{1}{2}(\mathbf{x}_{ni}-\mu_{i})\Sigma^{-1}_{ip}\Sigma^{-1}_{ql}(x_{nl}-\mu_{l})
        \right]\\
    =&\sum_{n=1}^N\left[
        -\frac{1}{2}\Sigma^{-1}_{qp}
        -
        \frac{1}{2}(\mathbf{x}_{ni}-\mu_{i})\Sigma^{-1}_{ip}\Sigma^{-1}_{ql}(x_{nl}-\mu_{l})
        \right]\\
    =&-\frac{N}{2}\Sigma^{-1}_{qp}
    -\sum_{n=1}^N\left[
        \frac{1}{2}(\mathbf{x}_{ni}-\mu_{i})\Sigma^{-1}_{ip}\Sigma^{-1}_{ql}(x_{nl}-\mu_{l})
        \right]=0\\
    \implies&\Sigma^{-1}_{qp}=
    -\frac{1}{N}\sum_{n=1}^N\left[
        (\mathbf{x}_{ni}-\mu_{i})\Sigma^{-1}_{ip}\Sigma^{-1}_{ql}(x_{nl}-\mu_{l})
        \right]\\
    \implies&\Sigma_{aq}\Sigma^{-1}_{qp}\Sigma_{pb}=
    -\frac{1}{N}\sum_{n=1}^N\left[
        \Sigma_{aq}(\mathbf{x}_{ni}-\mu_{i})\Sigma^{-1}_{ip}\Sigma^{-1}_{ql}(x_{nl}-\mu_{l})\Sigma_{pb}
        \right]\\
    \implies&\Sigma_{ab}=
    -\frac{1}{N}\sum_{n=1}^N\left[
        (\mathbf{x}_{ni}-\mu_{i})\delta_{ib}\delta_{al}(x_{nl}-\mu_{l})
        \right]\\
    =&-\frac{1}{N}\sum_{n=1}^N
        (\mathbf{x}_{nb}-\mu_{b})(x_{na}-\mu_{a})\qquad\Box

\end{align*}$$

## 9.2.2 Cont'd
다시 식 9.19로 돌아오자. 위에서의 식을 이용하여 약간의 기교를 부리면 나름 쉽게 (안 그러면 계산이 상당히 복잡해진다) 9.19를 유도할 수 있다.

$$\begin{align*}
    &\frac{\partial}{\partial \Sigma_{kpq}}\ln\prod_{n=1}^N\sum_{k'=1}^K\pi_{k'}\mathcal{N}(\mathbf{x}_n|\boldsymbol{\mu}_{k'},\boldsymbol{\Sigma}_{k'})\\
    =&\sum_{n=1}^N
        \frac{\partial}{\partial \Sigma_{kpq}}
        \ln
        \sum_{k'=1}^K
            \pi_{k'}
            \mathcal{N}(
                    \mathbf{x}_n|\boldsymbol{\mu}_{k'},\boldsymbol{\Sigma}_{k'})\\
    =&\sum_{n=1}^N
        \cfrac{
            \cfrac{\partial}{\partial \Sigma_{kpq}}
            \pi_{k}
            \mathcal{N}(
                    \mathbf{x}_n|\boldsymbol{\mu}_{k},\boldsymbol{\Sigma}_{k})}
            {\sum_{k'=1}^K
                \pi_{k'}
                \mathcal{N}(
                \mathbf{x}_n|\boldsymbol{\mu}_{k'},\boldsymbol{\Sigma}_{k'})}\\
    =&\sum_{n=1}^N
        \cfrac{
            \pi_{k}
            \cfrac{\partial \ln\mathcal{N}(\cdot)}{\partial \Sigma_{kpq}}
            \cfrac{\partial}{\partial \ln\mathcal{N}(\cdot)}
            \mathcal{N}(
                    \mathbf{x}_n|\boldsymbol{\mu}_{k},\boldsymbol{\Sigma}_{k})}
            {\sum_{k'=1}^K
                \pi_{k'}
                \mathcal{N}(
                \mathbf{x}_n|\boldsymbol{\mu}_{k'},\boldsymbol{\Sigma}_{k'})}\\
    =&\sum_{n=1}^N
        \cfrac{
            \pi_{k}
            \left[
                -\cfrac{1}{2}\Sigma^{-1}_{kqp}
                -\cfrac{1}{2}(x_{ni}-\mu_{i})\Sigma^{-1}_{kip}\Sigma^{-1}_{kql}(x_{nl}-\mu_{l})
            \right]
            \mathcal{N}(
                    \mathbf{x}_n|\boldsymbol{\mu}_{k},\boldsymbol{\Sigma}_{k})}
            {\sum_{k'=1}^K
                \pi_{k'}
                \mathcal{N}(
                \mathbf{x}_n|\boldsymbol{\mu}_{k'},\boldsymbol{\Sigma}_{k'})}=0\\
    \implies&
    \Sigma^{-1}_{kqp}\sum_{n=1}^N
        \cfrac{\pi_{k}
            \mathcal{N}(
                    \mathbf{x}_n|\boldsymbol{\mu}_{k},\boldsymbol{\Sigma}_{k})}
            {\sum_{k'=1}^K
                \pi_{k'}
                \mathcal{N}(
                \mathbf{x}_n|\boldsymbol{\mu}_{k'},\boldsymbol{\Sigma}_{k'})}
    =
    \sum_{n=1}^N
        \cfrac{
            \pi_{k}
            \left[
                -(x_{ni}-\mu_{i})\Sigma^{-1}_{kip}\Sigma^{-1}_{kql}(x_{nl}-\mu_{l})
            \right]
            \mathcal{N}(
                    \mathbf{x}_n|\boldsymbol{\mu}_{k},\boldsymbol{\Sigma}_{k})}
            {\sum_{k'=1}^K
                \pi_{k'}
                \mathcal{N}(
                \mathbf{x}_n|\boldsymbol{\mu}_{k'},\boldsymbol{\Sigma}_{k'})}\\
    \implies&
    \Sigma_{kab}\sum_{n=1}^N
        \cfrac{\pi_{k}
            \mathcal{N}(
                    \mathbf{x}_n|\boldsymbol{\mu}_{k},\boldsymbol{\Sigma}_{k})}
            {\sum_{k'=1}^K
                \pi_{k'}
                \mathcal{N}(
                \mathbf{x}_n|\boldsymbol{\mu}_{k'},\boldsymbol{\Sigma}_{k'})}
    =
    \sum_{n=1}^N
        \cfrac{
            \pi_{k}
            \mathcal{N}(
                    \mathbf{x}_n|\boldsymbol{\mu}_{k},\boldsymbol{\Sigma}_{k})
            \left[
                -(x_{ni}-\mu_{i})\delta_{ib}\delta_{al}(x_{nl}-\mu_{l})
            \right]}
            {\sum_{k'=1}^K
                \pi_{k'}
                \mathcal{N}(
                \mathbf{x}_n|\boldsymbol{\mu}_{k'},\boldsymbol{\Sigma}_{k'})}\\
    \implies&
    \Sigma_{kab}\sum_{n=1}^N
        \gamma(z_{nk})
    =
    \sum_{n=1}^N
        \gamma(z_{nk})
        \left[
            -(x_{nb}-\mu_{b})(x_{na}-\mu_{a})
        \right]\qquad\because\ \gamma(z_{nk})=\cfrac{\pi_{k}
            \mathcal{N}(
                    \mathbf{x}_n|\boldsymbol{\mu}_{k},\boldsymbol{\Sigma}_{k})}
            {\sum_{k'=1}^K
                \pi_{k'}
                \mathcal{N}(
                \mathbf{x}_n|\boldsymbol{\mu}_{k'},\boldsymbol{\Sigma}_{k'})}\\
    \implies&
    \Sigma_{kab}
    =
    -
    \frac{1}{N_k}
    \sum_{n=1}^N
        \gamma(z_{nk})
        (x_{nb}-\mu_{b})(x_{na}-\mu_{a})
        \qquad\because\ \sum_{n=1}^N\gamma(z_{nk})=N_k\qquad\Box
\end{align*}$$

9.2.2 챕터에서는 이제 딱히 어려운 것은 없다. Lagrangian method를 알면 9.21은 계산을 하지 않아도 눈에 보일 것이다.

# 9.3.1

식이 전반적으로 깨끗해져서 대부분은 직관적으로 보이나, 9.39의 경우 약간 직관적이지 않다. 일단,  본문에서 제시하는 관측값과 잠재함수의 결합분포를 살펴보자.

$$\begin{align}
    p \left(
            \underline{x},\underline{z} \middle| \underline{\pi}, \underline{\underline{\mu}}, \underline{\underline{\underline{\Sigma}}}
            \right)
        =
        \prod_{k=1}^K\pi_k^{z_k}
            \mathcal{N}\left(
                \underline{x} | \underline{\mu}_k, \underline{\underline{\Sigma}}_k
            \right)^{z_k}    
\end{align}
$$

여기서 $z$를 주변화 하면,

$$\begin{align*}
    p \left(
            \underline{x} \middle| \underline{\pi}, \underline{\underline{\mu}}, \underline{\underline{\underline{\Sigma}}}
            \right)
        =&
        \sum_{\underline{z'}\in\{\mathbf{e}_1,\mathbf{e}_1,\ldots,\mathbf{e}_K\}}
            \prod_{k=1}^K
            \left[
                \pi_k
                \mathcal{N}\left(
                    \underline{x} \middle| \underline{\mu}_k, \underline{\underline{\Sigma}}_k
                \right)
            \right]^{z'_k}\\
        =&
        \sum_{l=1}^K
            \prod_{k=1}^K
            \left[
                \pi_k
                \mathcal{N}\left(
                    \underline{x} \middle| \underline{\mu}_k, \underline{\underline{\Sigma}}_k
                \right)
            \right]^{\delta_{lk}}\\
        =&
        \sum_{l=1}^K
            \left[
                \pi_l
                \mathcal{N}\left(
                    \underline{x} \middle| \underline{\mu}_l, \underline{\underline{\Sigma}}_l
                \right)
            \right]
\end{align*}
$$

이전 챕터에서 많이 보았던 주변화된 Gaussian mixture가 된다. 식 9.35는 단순히 식 $(4)$를 여러 개 곱한 것이다. 거기에 라그랑주 승수법을 적용하면 식 9.37를 도출할 수 있다. 그럼 이제 식 9.38은 조건부확률의 정의로부터 자연스럽게 도출할 수 있다.

$$
\begin{align*}
    p\left(
        \underline{\underline{z}}
        \middle|
        \underline{\underline{x}},
        \underline{\pi},
        \underline{\underline{\mu}},
        \underline{\underline{\underline{\Sigma}}}
    \right)
    =&
    \frac{
        p\left(\underline{\underline{x}},
            \underline{\underline{z}}
            \middle|
            \underline{\pi},
            \underline{\underline{\mu}},
            \underline{\underline{\underline{\Sigma}}}
        \right)
    }{  
        p\left(
            \underline{\underline{x}}
            \middle|
            \underline{\pi},
            \underline{\underline{\mu}},
            \underline{\underline{\underline{\Sigma}}}
        \right)
    }
\end{align*}
$$

이로부터 식 9.39를 전개해 보자. $\underline{z}$ 가 one-hot vector라는 점을 상기하자.

$$
\begin{align*}
    \mathbb{E}\left[z_{nk}
                    \middle|
                    \underline{\underline{x}},
                    \underline{\pi},
                    \underline{\underline{\mu}},
                    \underline{\underline{\underline{\Sigma}}}
                    \right]
    =&
    \sum_{z_{nk}\in\{0,1\}}
        z_{nk}
        \,
        p\left(
            z_{nk}
            \middle|
            \underline{\underline{x}},
            \underline{\pi},
            \underline{\underline{\mu}},
            \underline{\underline{\underline{\Sigma}}}
            \right)\\
    =&
    \sum_{z_{nk}\in\{0,1\}}
        z_{nk}
        \frac{
            p\left(\underline{x}_n,
                \underline{z}_{n}
                \middle|
                \underline{\pi},
                \underline{\underline{\mu}},
                \underline{\underline{\underline{\Sigma}}}
            \right)
        }{  
            p\left(
                \underline{x}_n
                \middle|
                \underline{\pi},
                \underline{\underline{\mu}},
                \underline{\underline{\underline{\Sigma}}}
            \right)
        }\qquad\because\ \text{Assuming i.i.d. of }\underline{\underline{x}}.\\
    =&
    \sum_{z_{nk}\in\{0,1\}}
        z_{nk}
        \frac{
            \prod_{k'=1}^K
            \left(
                \pi_{k'}
                \,
                \mathcal{N}
                \left(
                    \underline{x}_n
                    \middle|
                    \underline{\mu}_{k'},
                    \underline{\underline{\Sigma}}_{k'}
                \right)
            \right)^{z_{nk'}}
        }{  
            \sum_{k'=1}^K
                \pi_{k'}
                \,
                \mathcal{N}\left(
                    \underline{x}_n
                    \middle|
                    \underline{\mu}_{k'},
                    \underline{\underline{\Sigma}}_{k'}
                    \right)
        }\\
    =&
    \frac{
        \pi_{k}
        \,
        \mathcal{N}
        \left(
            \underline{x}_n
            \middle|
            \underline{\mu}_{k},
            \underline{\underline{\Sigma}}_{k}
        \right)
    }{  
        \sum_{k'=1}^K
            \pi_{k'}
            \,
            \mathcal{N}\left(
                \underline{x}_n
                \middle|
                \underline{\mu}_{k'},
                \underline{\underline{\Sigma}}_{k'}
                \right)
    }\\
    &\qquad
    \because
    \ 
    z_{nk}=1\implies \forall\ k'\ne k:z_{nk'}=0\ \text{(one-hot vector)}
    \\
    =&\gamma(z_{nk})
\end{align*}
$$

마찬가지로 식 9.40도 유도할 수 있다. 책 본문에서는 간략하게 언급하여 혼동의 여지가 있는데, 이 기댓값에서 확률변수는 $\underline{\underline{z}}$ 뿐이다. 즉, $\underline{\underline{x}}$를 이용해 가능도의 기댓값을 추정하는 것이다.

$$
\begin{align*}
    \mathbb{E}_{\underline{\underline{Z}}}\left[
        \ln p
        \left(
            \underline{\underline{x}},
            \underline{\underline{Z}}
            \middle|
            \underline{\pi},
            \underline{\underline{\mu}},
            \underline{\underline{\underline{\Sigma}}}
        \right)
        \middle|
        \underline{\underline{x}},
        \underline{\pi},
        \underline{\underline{\mu}},
        \underline{\underline{\underline{\Sigma}}}
    \right]
    =&
    \mathbb{E}_{\underline{\underline{Z}}}\left[
        \sum_{n=1}^N
            \sum_{k=1}^K
                Z_{nk}
                \left[
                    \ln
                    \pi_k
                    \mathcal{N}\left(
                        \underline{x}_n
                        \middle|
                        \underline{\mu}_k,
                        \underline{\underline{\Sigma}}_k
                    \right)
                \right]
    \right]
    \\
    =&
    \sum_{n=1}^N
        \sum_{k=1}^K    
    \mathbb{E}_{\underline{\underline{Z}}}\left[
        Z_{nk}
        \ln
        \pi_k
        \mathcal{N}\left(
            \underline{x}_n
            \middle|
            \underline{\mu}_k,
            \underline{\underline{\Sigma}}_k
        \right)
    \right]
\end{align*}
$$

여기서 기댓값 안 $\ln$ 내부 값은 $Z_{nk}$가 어떤 것으로 선택되든 상관 없이 항상 고정되어 있다. 따라서 기댓값 밖으로 나올 수 있고, 이는 곧 식 9.40이 된다.

지금까지의 내용을 정리하면, Expectation 단계에서는 데이터셋과 파라미터를 이용해 잠재 변수의 조건부 분포를 계산한다. 그 후 Maximization 단계에서는 이 조건부 분포를 이용해 가능도의 기댓값을 만들어내고, (위 식에서의 기댓값 항에 $Z$의 $X$가 주어졌을 때의 확률이 필요하다는 점을 기억하자. 이것은 Expectation 단계에서 계산한 분포이다.) Lagrangian method와 같은 방법으로 해당 기댓값을 최대화하는 파라미터를 찾아낸다.

# 9.3.2

식 9.42에서 대충 말로만 얘기하고 넘어 가는 것이 그렇게 마음에 들지는 않아서, 직접 계산해봤다. $k$ 점 $\mathbf{x}_n$에 가장 가까운 점이라 가정한다.

$$
\begin{align*}
    \lim_{\epsilon\to0+}
    \gamma(z_{nk})
    =&
    \lim_{\epsilon\to0+}
    \frac{
        \pi_k\exp(-\|\mathbf{x}_n-\boldsymbol{\mu}_k\|^2/2\epsilon)
    }{
        \sum_{j=1}^K
            \pi_j\exp(-\|\mathbf{x}_n-\boldsymbol{\mu}_j\|^2/2\epsilon)
    }\\
    =&
    \lim_{\epsilon\to0+}
    \cfrac{
        1
    }{
        1
        +
        \sum_{j\ne k}
            \cfrac{\pi_j}{\pi_k}
            \exp
            \left[
                -
                \cfrac{
                    \|\mathbf{x}_n-\boldsymbol{\mu}_k\|^2-\|\mathbf{x}_n-\boldsymbol{\mu}_j\|^2 
                }{
                    2\epsilon
                }   
            \right]
    }\\
    =&\frac11\\
    =&1.
\end{align*}
$$

따라서 가장 가까운 클러스터의 책임값이 1이 되게 된다. 이제 $\pi$의 값은 어떤 값이 되든 0만 아니라면 영향이 없다. 이 책임값을 식 9.40에 넣고 전개하면 손쉽게 9.43을 얻을 수 있다. 보다시피 분산을 0으로 보내기 때문에, K-means 클러스터링은 분산을 고려할 수 없다.


*계속 작성 중.*