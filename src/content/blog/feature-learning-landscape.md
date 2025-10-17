---
title: All feature learning networks are/become balanced
pubDate: "2025-10-13"
description: "Feature learning made simple"
series: "Feature Learning"
seriesOrder: 1
draft: True
onlylink: True
---

I've been going through the literature on feature-learning linear networks, with the hope that it be useful to non-linear networks. Along the way, I came across "[From Lazy to Rich: Exact Learning Dynamics in Deep Linear Networks](https://arxiv.org/pdf/2409.14623)"\citep{dominé2025lazyrichexactlearning}, and while reading through, it seemed like they had identified a different type of feature learning than that of the maximum-update parameterization ($\mu$P) feature learning that I knew.

In this post, I want to discuss how the ideas of $\mu$P align with those of deep linear network balancedness in a unified way, and use new perspective to make predictions for how other 'ML trickery' that is often used in practice promotes feature learning.

### Setup

Unless otherwise stated, all networks will be 2 layer (1 hidden layer) networks of the form $\mathbf{f}(x) = \sum_{i} \mathbf{a}_i \sigma(\sum_jW_{ij}x_j)$. The input dimension of $d$ ($x \in \mathbb{R}^{d}$) and output dimension of $d_{out}$ ($f(x)\,, y \in \mathbb{R}^{d_{out}}$). The nonlinearity is to be taken as the identity function $\sigma: x\mapsto x$. The hidden dimension $w$, $W \in \mathbb{R}^{w \times d}$ is a parameter we can scale, where we wish for any dynamics to be consistent as $w$ is varied. The network is to be updated iteratively using gradient flow (without addons such as weight decay/momentum/etc.) $\Delta \mathbf{a} = -\eta_a \frac{\partial \mathcal{L}}{\partial \mathbf{a}}$, $\Delta W = -\eta_W \frac{\partial \mathcal{L}}{\partial W}$, ($\eta_a, \eta_W \ll 1$) under a square loss $\mathcal{L} = (f(x)-y)^2$ [^bsz].

## How to achieve feature learning

### $\mu$P

Based on \citep{yang:2023-spectral-scaling}, we should set up our network with the following conditions:

$$
\begin{align*}
    \sigma_W &= \Theta\left(\sqrt{\frac{1}{d}}\right)\\
    \sigma_a &= \Theta\left(\sqrt{\frac{d_{out}}{w^2}}\right)\\
    \eta_W &= \Theta\left(\frac{w}{d}\right)\\
    \eta_a &= \Theta\left(\frac{d_{out}}{w}\right)
\end{align*}
$$
where $a_i \sim \mathcal{N}(0, \sigma_a^2)$, $W_{i,j} \sim \mathcal {N}(0, \sigma_W^2)$. The broad intuition for this setup is such that we have (1) non-trivial updates to our weights (more specifically, the first weight matrix W), else we end up with a NNGP, and (2) not have our outputs blow up during training. Up to moving around the factors (à la mean-field parameterization), this has been identified as the only ($d$, $w$, $d_{out}$)-unique scaling of our network parameters.

### Balancedness

If that is true, then why does \citep{dominé2025lazyrichexactlearning} have a different setup that is removed from the $\mu$P scaling? The only condition they require[^balancednessreqs], which has been slightly modified to accomidate the layerwise learning rates, is that the network starts with

<div id="eq1">

$$
\begin{equation}
    \tau_a\mathbf{a}^\top \mathbf{a} - \tau_W WW^\top = \lambda I_w
\end{equation}
$$
</div>

where $\tau_{a/W} \equiv 1/(\eta_{a/W})$. This quantity will remain invariant over time, first reviewed in ["Get rich quick"]\citep{kunin2024richquickexactsolutions}, with a quick proof found [below](#conserved). With Gaussian initialization, this conserved equation is trivially achieved when the width w is large, making

$$
\begin{equation*}
    % \mathbf{a}^\top\mathbf{a} &\sim d_{out}\sigma_a^2 I_w\\
    % WW^\top &\sim d\sigma_W^2 I_w\\
    \tau_a\mathbf{a}^\top\mathbf{a}-\tau_WWW^\top=\left(\tau_a d_{out}\sigma_a^2-\tau_W d\sigma_W^2\right)I_w \equiv \lambda I_w,
\end{equation*}
$$

where $\sim$ is used due to sub-leading order corrections not being included. Thus far, we haven't needed to worry about making sure update sizes won't cause trivial or blown-up updates at all due to the trajectory of our network lying on an invariant manifold defined by [Eq. 1](#eq1), so what gives? The trick is in the value that $\lambda$: the closer to 0, the more our solutions appear to be in the feature-learning regime.

While our solution space for feature learning is quite large, with 3 different independent knobs to turn, if we rewrite our feature learning condition in the following way,

$$
\begin{equation*}
    \lambda = \tau_a d_{out}\sigma_a^2 - \tau_W d\sigma_W^2 = \frac{d_{out}}{\eta_a}\sigma_a^2 - \frac{d}{\eta_W}\sigma_W^2 \leq \epsilon
\end{equation*}
$$

for some small $\epsilon$ that'll represent how close to the "true" feature learning network we get, we can see a 'natural choice' for how we should scale our hyperparameters; taking both portions of the equation to be invariant to our network parameters of $(d, d_{out}, w)$ forces $\sigma_a^2 \sim \eta_a/d_{out}$, $\sigma_W^2 \sim \eta_W/d$, representing a perfect match with how \citep{yang:2023-spectral-scaling} told us to scale, up to actually choosing the layerwise learning rates and reintroducing a factor of $w$ which I believe has to do with the Gram matrix size being w x w, Jamie/Dhruva correct me if wrong.

For completeness, let's plug in the $\mu$P scaling laws to see how small $\lambda$ is:

$$
\begin{equation*}
    \lambda = \Theta\left(\frac{w}{d_{out}}d_{out}\frac{d_{out}}{w^2}-\frac{d}{w}d\frac{1}{d}\right)=\Theta\left(\frac{d_{out}}{w}-\frac{d}{w}\right)=\Theta \left(\frac{d_{out}-d}{w}\right)
\end{equation*}
$$

which tends towards 0 as $w\rightarrow \infty$.

### Lazy & ultra-rich scaling

On first glance, we have more freedom in the balancedness picture than the $\mu$P picture, so what gives? Lazy networks have thus far been great for theory, so surely balancedness is the way to go. Why are people so much more drawn to $\mu$P, outside of it being the first to introduce many of the feature learning concepts we know today?

Taking from the MFP literature, if we scale our network outputs as $ f \mapsto f/\gamma$ and the overall learning rate as $\eta \mapsto \eta\cdot\gamma^2$ for $\gamma \leq 1$ and $\eta \mapsto \eta\cdot\gamma$ otherwise, a scaling law discovered in \citep{atanasov:2024-ultrarich}, we can transition a network into the lazy regime with $\gamma \ll 1$ and into a new 'hyper-rich' regime of $\gamma \gg 1$. In the balancedness picture, both the network output prefactor and learning rate rescaling can be absorbed globally; it's unclear where to put the network output prefactor, but the learning rate can be easily included: 

$$
\lambda \sim
\begin{cases}
  \left(\tau_a d_{out}\sigma_a^2-\tau_W d\sigma_W^2\right)/\gamma^2, & \gamma \leq 1\\
  \left(\tau_a d_{out}\sigma_a^2-\tau_W d\sigma_W^2\right)/\gamma, & \gamma \text{ > } 1.
\end{cases}
$$

plugging in our $\mu$P relations, this gets

$$
\lambda(\gamma) = \Theta\left(
\begin{cases}
    \frac{d_{out}-d}{w\gamma^2}, & \gamma \leq 1\\
    \frac{d_{out}-d}{w\gamma}, & \gamma \text { > } 1
\end{cases}
\right),
$$

which clearly recovers $\lambda \rightarrow 0 (\pm\infty)$ as $\gamma \rightarrow \infty(-\infty)$.

This suggests the balancedness can be used in conjunction with the $\mu$P scaling in order to gain a deeper understanding of feature learning. 

## Big picture: why (insert ML trick) lets you learn features

Up until now, we've considered an extremely simple network with no special bells or whistles. If we want to understand how different ML levers affect feature learning, we need to move beyond this, and now that we have the balancedness-$\mu$P picture, we have a greater capacity to add on extra parts.

### Weight decay

Take (L2) weight decay, for example: the purpose is to decrease the weights of a network. Previous papers [links] have investigated ...

There doesn't exist an invariant when training is done with weight decay (except under the explicit case that the layerwise learning rates are identical and additionally the weight decay constant is global), but we can see how our earlier constant $\lambda$ varies with the weight penalty hyperparameter $\zeta$:

$$
\begin{equation*}
    \frac{\text{d}\lambda}{\text{d}t} = -2\zeta\left(\lvert\lvert a\rvert\rvert_F-\lvert\lvert W\rvert\rvert_F\right)
\end{equation*}
$$

where we note that weight decay lowers the overall scale of weights by forcing the weights between layers to become balanced, forcing the network parameters onto a 'more feature-learning manifold'.

### Learning rate warmup

To analyze learning rate increases (and subsequent decays), we need to move away from gradient flow. We'll still let $\lambda = \lvert\lvert a\rvert\rvert_F^2/\eta_a-\lvert\lvert W\rvert\rvert_F^2/\eta_W$ and now have $\Delta\lambda = \eta_a\lvert\lvert R(WX)^\top\rvert\rvert_F^2-\eta_W\lvert\lvert a^\top R X^\top\rvert\rvert_F^2$ for residual $R \equiv Y-aWX$. If we define $M = XR^\top$, we can rewrite $\Delta \lambda$ as,

$$
\begin{align*}
    \Delta \lambda(t) &= \eta_a(t) \alpha_t - \eta_W(t)\beta_t \leq -\eta_a\eta_W \lvert\lvert M\rvert\rvert_F^2 \lambda\\
    \alpha_t &= \lvert\lvert WM\rvert\rvert_F^2\\
    \beta_t &= \lvert\lvert a^\top M\rvert\rvert_F^2
\end{align*}
$$

with the general interpretation being for any initial values $\lambda$ far from 0, learning up the learning rate allows the weights to take much faster progress towards $\lambda=0$; this needs to be followed up by a decrease in the learning rate, or else $\lambda$ will end up "bouncing" between the positive and negative of some final value $\pm \lambda_f$.


### Deeper network balancedness

If we want to consider deeper networks, the story is largely the same. Take $Y=W_LW_{L-1}\cdots W_2W_1X$. Between two subsequent layers, we have

$$
\begin{equation*}
\tau_{\ell+1}W_{\ell+1}^\top W_{\ell+1}-\tau_{\ell+1}W_{\ell} W_{\ell}^\top = \lambda_\ell I_{w_\ell}
\end{equation*}
$$ 

that is, $W_{\ell+1}^\top W_{\ell+1}$ and $W_{\ell} W_{\ell}^\top$ are simultaneously diagonalizable, giving

$$
\begin{equation*}
\sigma_i(W_{\ell+1})^2= \sigma_i(W_{\ell})^2+\lambda_\ell
\end{equation*}
$$

while we can't generally compare $\tau_i W_{i}^\top W_{i} - \tau_j W_{j} W_{j}^\top$ ($j \neq i-1$) due to possible shape inconsistency, we can say

$$
\begin{equation*}
\sigma_i(W_{\ell+1})^2= \sigma_i(W_{k})^2+\sum_{i=k}^{\ell-1}\lambda_i
\end{equation*}
$$

with the implication being that there is a general balancedness between all layers that feature learning networks satisfy: the $i$-th singular vectors of all layers in a feature learning network are all roughly equivalent.

## Appendix

<div id='conserved'>

### Conserved quantity 

Write later

</div>

## References


[^bsz]: It should be mentioned that in my experiments, I employ a batch size, although I suspect this minimally changes results.

[^balancednessreqs]: The paper itself details two other requirements: white data, and non-bottleneckedness/non-overparameterized. While the first condition can be trivially dropped for our discussion, we always assume $w$ > $\max(d, d_{out})$; from what I can tell, the non-overfitting condition can be dropped for our discussion.