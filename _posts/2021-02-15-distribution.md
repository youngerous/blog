---
title: "Probability Distributions"
categories: 
  - study
last_modified_at: 2021-02-15
tags:
  - probability distribution
toc: true
toc_sticky: true
use_math: true
---

# Overview

<img src="/note/assets/figures/distribution.png" width="80%">


자세한 수식 전개는 생략하고 기본적인 개념과 분포 간 관계에 대해 살펴본다.

# 1. 범주형 확률분포 

## 1-1. 베르누이분포
0 또는 1의 경우만 존재하는 상황에서의 확률분포이다. 성공/실패, 앞면/뒷면 등의 예시가 있다.

$$ f_x(x;p) = p^x(1-p)^{1-x} $$


$$ x=0 \; or \; 1 $$

### 파라미터
- $p$: 성공 확률

### 기댓값

$$ E[x] = \sum_{x=0,1} x\cdot p^x (1-p)^{1-x} = 0+p=p $$

### 분산

$$ V[x] = E[X^2]-\{E[X]\}^2 = p(1-p)$$

## 1-2. 이항분포
베르누이 시행을 **독립적으로 $n$번 수행**했을 때의 확률분포이다.

$${ p(x) = {n \choose x} p^x (1-p)^{1-x} }$$ 


$${for \; x=0,1,\cdots,n }$$

### 파라미터
- $n$: 베르누이 시행 횟수
- $p$: 성공 확률

### 기댓값

$${  E[X] = np }$$

### 분산

$${ V[X]=np(1-p)}$$

## 1-3. 포아송분포
<img src="/note/assets/figures/poisson.jpeg" width="80%">


**단위 시간 내에 특정 사건이 몇 번 발생**할 것인지를 설명하는 확률분포이다.

$${ P\{ X=i \}} = { {e^{- \lambda} \lambda^i} \over i ! } $$

$${ i=0,1,2,\cdots }$$

이항분포에서 $n$이 매우 크고 $p$가 충분히 작다면 포아송분포로 근사할 수 있다.

### 파라미터
- $\lambda$: 단위 시간 당 사건 발생 횟수

### 기댓값

$${E[X]=\lambda }$$

### 분산

$${V[X]=\lambda }$$

## 1-4. 기하분포

<img src="/note/assets/figures/geometric.png" width="80%">


베르누이 시행에서 **첫 번째 성공**이 발생할 때까지 필요한 시행 횟수를 설명하는 확률분포이다. $n-1$번의 실패 후 성공하는 상황을 확률질량함수로 표현할 수 있다.

$${ \sum_{n=1}^{\infty} P\{X=n\} = p \sum_{n=1}^\infty (1-p)^{n-1} = {p \over 1-(1-p)}=1 }$$

### 파라미터
- $p$: 성공 확률


### 기댓값

$${E[X] = {1\over p} }$$

### 분산

$${V[X] = {1-p \over p^2} }$$

## 1-5. 음이항분포

<img src="/note/assets/figures/negbinomial.jpeg" width="80%">


기하분포의 일반화된 형태로서, 베르누이 시행에서 **$r$번째 성공**이 일어날 때까지 필요한 시행 횟수를 설명하는 확률분포이다.
$n-1$번의 시행까지 $r-1$번의 성공이 존재해야 하는 상황을 확률밀도함수로 표현할 수 있다.

$${ P\{ X=n\} = {n-1 \choose r-1} p^r (1-p)^{n-r}}$$


$${ n=r, r+1, \cdots}$$


$r=1$이면 기하분포와 같은 형태이다.

### 파라미터
- $r$: 성공 횟수
- $p$: 성공 확률

### 기댓값

$${ E[X] = {r \over p} }$$

### 분산

$${ V[X] = {r(1-p) \over p^2} }$$

## 1-6. 초기하분포

<img src="/note/assets/figures/hypergeometric.jpeg" width="80%">


$N$개의 공 가운데 $m$개가 하얀색이고 $N-m$개가 빨간색인 경우, **비복원추출**로 $n$개의 공을 뽑을 때 $i$개가 하얀색인 확률분포를 나타낸다. 불량품 검출 상황에서도 자주 사용된다.

$${ P\{X=i\} = { {m \choose i} {N-m \choose n-i} \over {N \choose n}} }$$

$${ i=0,1,\cdots,n }$$

### 파라미터
- $N$: 대상의 전체 개수
- $m$: 특정 상태를 갖는 대상의 개수
- $n$: 비복원추출 개수

### 기댓값

$${ E[X] = {nm \over N}  \left[ {(n-1)(m-1) \over N-1} + 1 \right] }$$

### 분산

$${ V[X] = np(1-p)\left( 1- {n-1 \over N-1} \right)}, \;\; where \; p = {m \over N} $$

# 2. 연속형 확률분포

## 2-1. 일양(Uniform)분포

### PDF

<img src="/note/assets/figures/uniformpdf.jpeg" width="80%">


$${ f(x)=
\begin{cases}
{1 \over \beta-\alpha}, & {if \; \alpha < x < \beta} \\
0, & \text{otherwise}
\end{cases} 
}$$

### CDF

<img src="/note/assets/figures/uniformcdf.jpeg" width="80%">

$${ F(a) = \int_{-\infty}^{a} f(x) \, dx  \;\; , \text{X is uniform on } (\alpha, \beta)  }$$

**① $ a \le \alpha $** <br>

$${ F(a) = 0 }$$

**② $ \alpha < a < \beta $** <br>

$${ F(a) = {a-\alpha \over \beta - \alpha} }$$

**① $ \beta \le a $** <br>

$${ F(a) = 1 }$$


### 기댓값

$${ E[X] = {\beta + \alpha \over 2} }$$

### 분산

$${ V[X] = {(\beta - \alpha)^2 \over 12} }$$

## 2-2. 정규분포

<img src="/note/assets/figures/normal.jpeg" width="80%">


평균을 기준으로 대칭 형태이며, unimodal인 경우 평균에서 가장 높은 확률을 갖는다.

$${ f(x; \mu, \sigma) = {1 \over \sigma \sqrt{2 \pi} } e^{-{1 \over 2} \left( {x - \mu} \over \sigma \right)^2} }$$

$${ -\infty < x < \infty }$$

### 파라미터
- $\mu$: 평균
- $\sigma$: 표준편차

### 기댓값

$$ E[X] = \mu $$

### 분산

$$ V[X] = \sigma^2 $$


표준정규분포의 확률변수 $ Z = {X-\mu \over \sigma}  $ 는 $\mu=0, \sigma=1$의 파라미터를 갖는다. 표준정규분포의 CDF는 표를 보고 확인할 수 있다. 


또한 정규분포는 이항분포로 근사할 수 있다. CDF 계산이 복잡한 (이산형의) 이항분포를 연속형으로 근사하여 편하게 계산하려는 목적이다. 다만 아래와 같은 조건이 만족되어야 한다.

- $n$이 충분히 커야 한다.
- $p$가 0 또는 1에 너무 가깝지 않아야 한다.

조건을 만족한다면 아래와 같이 표현할 수 있다.

<img src="/note/assets/figures/normalapprox.jpeg" width="80%">


$${ X\sim Binomial(n,p) \approx Y\sim Normal(np, np(1-p)) }$$

## 2-3. 지수분포

<img src="/note/assets/figures/exp.png" width="80%">


포아송분포가 단위시간동안 어떤 사건이 발생한 횟수에 대한 확률분포라면, 지수분포는 **포아송분포에서 발생하는 사건 사이의 시간**에 대한 확률분포이다. 따라서 어떤 사건이 포아송분포에서 발생했다면 사건 사이의 시간은 지수분포를 따를 것이고, 사건 사이의 시간이 지수분포를 따른다면 사건 발생 횟수는 포아송분포를 따를 것이다.


### PDF

$${f(x)=
\begin{cases}
\lambda e^{-\lambda x}, & \text{if } x\ge 0 \\
0, & \text{if } x<0
\end{cases}
}$$

### CDF

$${F(x))=
\begin{cases}
0, & \text{if } x<0 \\
1 -  e^{-\lambda x}, & \text{if } x\ge 0 
\end{cases}
}$$

### 파라미터
- $\lambda$: 단위 시간 당 발생하는 사건의 평균 횟수

### 기댓값

$${ E[X] = {1 \over \lambda} }$$

### 분산

$${ V[X] = {1 \over \lambda^2} }$$


지수분포는 **Memoryless Property**를 갖는다. $t$라는 시간만큼 지난 과거의 상태는 앞으로 고려할 시간 $s$에 영향을 미치지 못한다는 뜻이다. 예를 들어, 어떤 기계가 $t$ 시간동안 작동한 상태에서 $s$ 시간만큼 더 작동할 확률은 단순히 $s$ 시간 더 작동할 확률과 같다는 뜻이다.




## 2-4. 감마분포

<img src="/note/assets/figures/gamma.png" width="80%">


지수분포의 일반화된 형태이다. 즉 $\alpha$개의 사건이 발생할 때까지의 시간을 다루는 확률분포이다. 

$${f(x)=
\begin{cases}
{\lambda \over \Gamma{(\alpha)}} x^{\alpha-1}e^{-\lambda x}, & \text{if } x > 0 \\
0, & \text{otherwise}
\end{cases}
}$$


$${ \Gamma{(\alpha)} = \int_{0}^{\infty} x^{\alpha-1}e^x \, dx }$$

### 파라미터
- $\alpha$: 사건 발생 횟수 (shape parameter)
- $\lambda$: 단위시간 당 발생하는 사건의 평균 횟수 (rate parameter)

### 기댓값

$${ E[X] = {\alpha \over \lambda} }$$

### 분산

$${ V[X] = {\alpha \over \lambda^2} }$$


$X_1, \cdots, X_\alpha \sim_{iid} Exponential(\lambda) $ 일 경우, $X_1, \cdots, X_\alpha$는 $\alpha, \lambda$를 파라미터롤 갖는 감마분포를 따른다. 즉 감마분포는 지수분포의 합으로 표현할 수 있다. 정리하자면 단위시간 당 사건발생횟수가 $\lambda$인 포아송분포에서 $\alpha$개의 사건이 발생할 때까지 시간은 감마분포를 따른다.

## 2-5. 베타분포

<img src="/note/assets/figures/beta.png" width="80%">


비율을 설명할 때 사용하는 확률분포이다.

$${f(x)=
\begin{cases}
{1 \over B{(\alpha, \beta)}} x^{\alpha-1} (1-x)^{\beta -1}, & \text{for } 0 \le x \le 1 \\
0, & \text{otherwise}
\end{cases}
}$$


$${ B(\alpha, \beta) = {\Gamma(\alpha) \Gamma(\beta) \over \Gamma(\alpha + \beta)} = \int_0^1 x^{\alpha-1} (1-x)^{\beta-1} }$$

### 파라미터
- $\alpha$, $\beta$

### 기댓값

$${ E[X] = {\alpha \over \alpha + \beta} }$$

### 분산

$${ V[X] = {\alpha \beta \over (\alpha + \beta)^2 (\alpha + \beta + 1)} }$$

# 요약: 확률분포 간 관계

<img src="/note/assets/figures/distribution.png" width="80%">


**베르누이분포**를 독립적으로 $n$번 수행하면 **이항분포**를 따른다. 이항분포는 샘플이 충분히 많고 성공 확률이 0 또는 1에 너무 가깝지 않다는 조건 하에서 **정규분포**로 근사할 수 있다.


**기하분포**는 베르누이 시행에서 첫 번째 성공이 일어날 때까지 필요한 시행 횟수를 설명한다. 이를 일반화하여 $k$번째 성공이 일어날 때까지 필요한 시행 횟수를 설명하는 분포는 **음이항분포**이다. 한편 **초기하분포**는 두 가지 상태(이항분포)를 갖는 전체 샘플에서 비복원추출로 몇 개를 추출할 때 특정 상태를 갖는 샘플의 갯수를 설명한다.


**포아송분포**는 단위시간 내에 특정 사건이 몇 번 발생할 것인지를 표현하는데, 이항분포에서 수행 횟수가 매우 크고 성공 확률이 매우 낮다면 포아송분포로 근사할 수 있다. 포아송분포에서 발생한 사건 사이의 시간을 표현할 때에는 **지수분포**를 사용한다. 지수분포를 일반화하여 포아송분포에서 발생한 $k$개의 사건이 발생할 때까지의 시간을 표현할 때에는 **감마분포**를 사용한다. 지수분포에서 독립적으로 샘플링된 확률변수의 합은 감마분포를 따른다. 한편 **베타분포**는 주로 고장률 등의 비율을 설명하는 데 사용되고, 베타분포 내 베타함수는 감마함수로 표현된다. 



# 참고자료
- [김성범 교수님 - 핵심 확률/통계](https://www.youtube.com/playlist?list=PLpIPLT0Pf7IqS4as3nefPyGv94r2aY6IT)