### 1、背景知识
#### 1.1 一种随着季度变化的推荐系统
假设有一个用户-商品-季度构成的张量 $\mathcal{Y}$ ,大小为 $m \times n \times f$ ,其中, $m$ 表示用户的数量, $n$ 表示商品的数量, $f$ 表示总共的季度,那么,将用户因子 $u_{is}$ 和商品因子 $v_{js}$ 构成的矩阵分解($r$表示分解结构的秩)
$y_{ij} \approx \sum_{s=1}^r u_{is} v_{js}$
添加关于时间的因子,形式如下:
$y_{ijt} \approx u_{is} v_{js} x_{ts}$
考虑实际的数据往往存在随机性,假设高斯分布适用于该张量分解任务,即:
$y_{ijt} \sim \mathcal{N}(\sum_{s=1}^r u_{is} v_{js} x_{ts} ,\tau^{-1})$
- 什么是高斯分布?
高斯分布也称为正态分布,简写为 $\mathcal{N}$ ,其概率密度函数为:
$\large \mathcal{N}(x|\mu,\tau) = \sqrt{\frac{\tau}{2\pi}} e^{-\frac{1}{2} \tau (x-\mu)^2}$
其中,$\tau$ 是精度项,等于正态分布中方差的倒数,这样定义可以方便共轭先验的选取。
- 实际上在张量分解的高斯分布假设中,指数项 $- \frac{1}{2} \tau (y_{ijt} - \sum_{s=1}^r u_{is} v_{js} x_{ts})^2 $ 和张量分解常用的损失函数 $min_{U,V,X} \frac{1}{2} \sum_{(i,j,t) \in \Omega} (y_{ijt} - \sum_{s=1}^r u_{is} v_{js} x_{ts})^2$ 形式上是一致的,所不同的是,采用贝叶斯框架在求解过程中能够有效的应对非凸优化问题。
`在张量分解中,任务是寻求合理的分解结构,为了通过贝叶斯推断估计出因子矩阵` $U \in \mathbb{U}^{m \times r} ,\mathbb{V}^{n \times r},\mathbb{U}^{f \times r}$,需要进一步设置<u>共轭先验</u> (仍然是高斯分布)
$u_i \sim \mathcal{N}(\mu_u,\Lambda_u^{-1})$
$v_j \sim \mathcal{N}(\mu_v,\Lambda_v^{-1})$
顺便提一下，带有时序变化的推荐系统当然也有一些优势，举个简单的例子，如果我们找到了用户 $A$ 的偏好,发现他以往喜欢绿色的鞋子,但当季火爆的鞋子却是蓝色的,那么推荐系统这时候再推荐一款绿色的鞋子可能就不太符合用户 $A$ 的口味了。
为了把用户 $A$ 在上一个季度的偏好(比如青睐阿迪达斯)和本季度的商品推荐完美地结合起来,于是,一个巧妙的设计产生了,即:
$x_t \sim \mathcal{N}(x_{t-1},\Lambda_x^{-1}),t=2,3,\dots,f$
详见[1]

#### 1.2 贝叶斯张量分解与图像复原
详见[2]


### 2 贝叶斯高斯张量分解模型
#### 2.1 贝叶斯推断的基础知识
贝叶斯推断是求解贝叶斯张量分解的重要工具，如果我们不加深对贝叶斯推断的理解，那么掌握贝叶斯张量分解可能就变成了无稽之谈。
一般而言，贝叶斯推断主要分为随机型采样和确定型参数估计，代表性的算法分别有 $MCMC$ ($Markov~chain~Monte~Carlo$, 马尔可夫链蒙特卡洛)和变分推断，相比之下，变分推断要比$MCMC$ 抽象,在迭代过程中,每代的计算消耗也较大,不过其主要优势是得到模型近似解的收敛速度快。
#### 2.2 极简的贝叶斯高斯张量分解模型
贝叶斯高斯张量分解模型事实上是一种特殊的贝叶斯模型，在很多关于贝叶斯张量分解或者贝叶斯矩阵分解的研究中，给定高斯假设是最理想的做法，并且模型的复杂程度完全取决于共轭先验的设计，在这里，我们先来看一个极简的模型。
- 再次重写一下模型:
$y_{ijt} \sim \mathcal{N}(\sum_{s=1}^r u_{is} v_{js} x_{ts},\tau^{-1})$
- 模型参数的先验分布:
$u_i,v_j,x_t \sim \mathcal{N}(0,[diag(\lambda)]^{-1}),\forall i,j,t$
$\tau \sim Gamma(\alpha,\beta)$
- 超参数的先验分布:
$\lambda_s \sim Gamma(\alpha,\beta),s=1,2,\dots,r$
![BPCP](./image/BPCP.png)

##### 2.2.1 推导模型参数 $u_i$ 的后验分布
就模型参数 $u_i$ 而言,似然来自 $\mathcal{Y}_{:jt}$ 中被观测到的元素:
$\mathcal{L} (\mathcal{Y}_{:jt} | u_i,V,X,\tau) $
$\Large \propto \prod_{:,j,t} e^{- \frac{1}{2} \tau(y_{ijt} - u_i^Tw_{jt})^2} $
是很多高斯分布的乘积。其中 $u_i^T w_{jt} = \sum_{s=1}^r u_{is} v_{js} x_{ts} ,w_{jt} = (v_{j} \circledast x_t) ,\circledast $ 代表点乘。 
$\Large \propto \prod_{:,j,t} e^{- \frac{1}{2} \tau (y_{ijt} - u_i^Tw_{jt}) (y_{ijt} - u_i^Tw_{jt})^T} $
$\Large \propto e^{- \frac{1}{2} u_i^T (\tau \sum_{:,j,t} w_{jt} w_{jt}^T) u_i + \frac{1}{2} u_i^T (\tau \sum_{:,j,t} y_{ijt} w_{jt})}$
可以得到关于 $u_i$ 的多元正态分布
由于 $u_i \sim \mathcal{N}(0,[diag(\lambda)]^{-1}) $
$\Large p(u_i | \lambda) \propto e^{-\frac{1}{2} u_i^T diag(\lambda) u_i}$
根据贝叶斯准则 $posterior \propto prior \times likehood$
$p(u_i | V,X,\tau,\mathcal{Y}_{:j,t},\lambda) \propto p(u_i | \lambda) \times \mathcal{L}(\mathcal{Y}_{:j,t} |u_i,V,X,\tau)$ 
$\Large \propto e^{-\frac{1}{2} u_i^T diag(\lambda) u_i} e^{- \frac{1}{2} u_i^T (\tau \sum_{:,j,t} w_{jt} w_{jt}^T) u_i + \frac{1}{2} u_i^T (\tau \sum_{:,j,t} y_{ijt} w_{jt})}$
$\Large \propto e^{-\frac{1}{2} u_i^T [diag(\lambda) + \tau \sum_{:,j,t} w_{jt} w_{jt}^T] u_i} e^{\frac{1}{2} u_i^T (\tau \sum_{:,j,t} y_{ijt} w_{jt})}$
令 $\widetilde{\Lambda}_u = diag(\lambda) + \tau \sum_{:,j,t} w_{jt} w_{jt}^T$
$\Large \propto e^{-\frac{1}{2} u_i^T \widetilde{\Lambda}_u u_i + \frac{1}{2} u_i^T (\tau \sum_{:,j,t} y_{ijt} w_{jt})}$
$\Large \propto e^{-\frac{1}{2} (u_i - \widetilde{u}_u)^T \widetilde{\Lambda}_u (u_i - \widetilde{u}_u)}$
其中 $\widetilde{u}_u = \tau \widetilde{\Lambda}_u^{-1} \sum_{:,j,t} y_{ijt} w_{jt}$
即: $u_i \sim \mathcal{N}(\widetilde{u}_u,\widetilde{\Lambda}_u^{-1})$

##### 2.2.2 推导模型参数 $v_j$ 的后验分布
与 $u_i$ 的推导类似
$\widetilde{\Lambda}_v = diag(\lambda) + \tau \sum_{i,:,t} w_{it} w_{it}^T$
其中 $\widetilde{v}_v = \tau \widetilde{\Lambda}_v^{-1} \sum_{i,:,t} y_{ijt} w_{it}$
即: $v_j \sim \mathcal{N}(\widetilde{v}_v,\widetilde{\Lambda}_v^{-1})$
##### 2.2.3 推导模型参数 $x_t$ 的后验分布
与 $u_i$ 的推导类似
$\widetilde{\Lambda}_x = diag(\lambda) + \tau \sum_{i,j,:} w_{ij} w_{ij}^T$
其中 $\widetilde{x}_x = \tau \widetilde{\Lambda}_x^{-1} \sum_{i,j,:} y_{ijt} w_{ij}$
即: $x_t \sim \mathcal{N}(\widetilde{x}_x,\widetilde{\Lambda}_x^{-1})$
##### 2.2.4 推导模型参数 $\tau$ 的后验分布
已知先验 $\Large p(\tau | \alpha_0,\beta_0) =  p(\tau | \alpha_0,\beta_0) = \frac{(\beta_0)^{\alpha_0}}{\Gamma(\alpha_0)} (\tau)^{\alpha_0-1} e^{-\beta_0\tau}$
就模型参数 $\tau$ 而言,似然主要来自于 $\mathcal{Y}$
$\Large \mathcal{L}(\mathcal{Y} | \tau,U,V,X) \propto \prod_{i=1}^m \prod_{j=1}^n \prod_{t=1}^f \tau^{1/2} e^{-\frac{1}{2} \tau (y_{ijt} - \sum_{s=1}^r u_{is} v_{js} x_{ts})^2 }$
$\Large \propto \tau^{\frac{1}{2}(m + n + f)} e^{-\frac{1}{2} \tau \sum_{i,j,t \in \Omega}(y_{ijt} - \sum_{s=1}^r u_{is} v_{js} x_{ts})^2 }$
所以:$\large p(\tau|-) \propto \mathcal{L}(\mathcal{Y} | \tau,U,V,X) \times p(\tau | \alpha_0,\beta_0)$
$\Large \propto \tau^{\frac{1}{2}(m + n + f)} e^{-\frac{1}{2} \tau \sum_{i,j,t \in \Omega}(y_{ijt} - \sum_{s=1}^r u_{is} v_{js} x_{ts})^2 } (\tau)^{\alpha_0-1} e^{-\beta_0\tau}$
$\Large \propto \tau^{\frac{1}{2}(m + n + f) + \alpha_0 -1} e^{-\frac{1}{2} \tau \left[\beta_0 +  \sum_{i,j,t \in \Omega}(y_{ijt} - \sum_{s=1}^r u_{is} v_{js} x_{ts})^2 \right] }$


在张量分解的贝叶斯网络中,可以通过 $\tau \sim Gamma(\widetilde{\alpha},\widetilde{\beta})$ 对参数 $\tau$ 进行采样更新,其中:
$\widetilde{\alpha} = a_0 + \frac{1}{2} \sum_{i,j,t \in \Omega} 1 (y_{ijt} \neq 0)$
$\widetilde{\beta} = \beta_0 + \frac{1}{2} \sum_{i,j,t \in \Omega} (y_{ijt} - \sum_{s=1}^r u_{is} v_{js} x_{ts})^2 $

##### 2.2.5 推导模型参数 $\lambda$ 的后验分布
已知先验分布 $\Large p(\lambda_s | \alpha_0,\beta_0) =  p(\lambda_s | \alpha_0,\beta_0) = \frac{(\beta_0)^{\alpha_0}}{\Gamma(\alpha_0)} (\lambda_s)^{\alpha_0-1} e^{-\beta_0 \lambda_s}$
就模型参数 $\lambda_s$ 而言,其似然主要来自于 $U,V,X$
尽管超参数 $\lambda$ 与参数 $\tau$ 都被假设服从伽马分布,但不同的是,参数 $\lambda$ 作为一个向量,对应多元正态分布中的协方差矩阵,在这里,不妨以 $u_i$ 为例,先写一下多元正态分布的形式
$\Large p(u_i | \lambda) = \frac{|diag(\lambda)|^{1/2}}{(2\pi)^{r/2}} e^{-\frac{1}{2} u_i^T diag(\lambda) u_i}$
从这条公式中,对于任意 $\lambda_{s} ~ s=1,2,\dots,r$
$\Large p(u_{is}| \lambda_{s}) \propto (\lambda_{s})^{1/2} e^{-\frac{1}{2} \lambda_{s} u_{is}^2}$
所以 $\Large \mathcal{L}(U,V,X | \lambda_s) \propto \prod_{i=1}^m (\lambda_{s})^{1/2} e^{-\frac{1}{2} \lambda_{s} u_{is}^2} \prod_{j=1}^n (\lambda_{s})^{1/2} e^{-\frac{1}{2} \lambda_{s} v_{js}^2} \prod_{t=1}^f (\lambda_{s})^{1/2} e^{-\frac{1}{2} \lambda_{s} x_{ts}^2} $
$\Large \propto (\lambda_s)^{(m+n+f)} e^{-\frac{1}{2} \lambda_s \left[ \sum_{i=1}^m u_{is}^2 + \sum_{j=1}^n v_{js}^2 + \sum_{t=1}^f x_{ts}^2 \right]}$
所以 $\large p(\lambda_s | U,V,X,\alpha_0,\beta_0) = p(\lambda_s | \alpha_0,\beta_0) \times \mathcal{L}(U,V,X | \lambda_s)$
$\Large \propto (\lambda_s)^{(m+n+f)} e^{-\frac{1}{2} \lambda_s \left[ \sum_{i=1}^m u_{is}^2 + \sum_{j=1}^n v_{js}^2 + \sum_{t=1}^f x_{ts}^2 \right]} (\lambda_s)^{\alpha_0-1} e^{-\beta_0 \lambda_s}$
$\Large \propto (\lambda_s)^{(m+n+f) + \alpha_0-1} e^{- \lambda_s \left[\frac{1}{2} (\sum_{i=1}^m u_{is}^2 + \sum_{j=1}^n v_{js}^2 + \sum_{t=1}^f x_{ts}^2 )+ \beta_0 \right]}$

超参数 $\lambda_s \sim Gamma(\widetilde{\alpha},\widetilde{\beta}),s=1,2,\dots,r$
$\widetilde{\alpha} = \alpha_0 + \frac{1}{2} (m+n+f)$
$\widetilde{\beta} = \beta_0 + \frac{1}{2} (\sum_{i=1}^m u_{is}^2 + \sum_{j=1}^n v_{js}^2 + \sum_{t=1}^f x_{ts}^2)$


### 参考文献
- [1] [Xiong L, Chen X, Huang T K, et al. Temporal collaborative filtering with bayesian probabilistic tensor factorization[C]//Proceedings of the 2010 SIAM international conference on data mining. Society for Industrial and Applied Mathematics, 2010: 211-222.](https://www.cs.cmu.edu/~jgc/publication/PublicationPDF/Temporal_Collaborative_Filtering_With_Bayesian_Probabilidtic_Tensor_Factorization.pdf)
[论文代码](https://www.cs.cmu.edu/~lxiong/bptf/bptf.html)
- [2] [Zhao Q, Zhang L, Cichocki A. Bayesian CP factorization of incomplete tensors with automatic rank determination[J]. IEEE transactions on pattern analysis and machine intelligence, 2015, 37(9): 1751-1763.](https://ieeexplore.ieee.org/document/7010937)
[论文代码](https://ieeexplore.ieee.org/document/7010937/media#media)
