# ICLR2025中DPO(Direct Reference Optimization)相关文章收录

## 1. 3D-Properties: Identifying Challenges in DPO and Charting a Path Forward
3D 属性：识别 DPO 中的挑战并规划前进道路

### 关键字
* LLM (DPO主要用于大模型微调)
* DPO (直接偏好优化 Direct Preference Optimization)
* RLHF (Reinforcement Learning from Human Feedback)

### 摘要
Aligning large language models (LLMs) with human preferences has recently garnered significant attention, with Proximal Policy Optimization (PPO) being a canonical yet computationally expensive method, and Direct Preference Optimization (DPO) offering a simpler and more efficient alternative. While prior studies have explored the trade-offs between PPO and DPO, DPO remains underutilized in state-of-the-art production-level LLMs, suggesting potential limitations. In this work, we revisit DPO with a comprehensive analysis of its theoretical foundations and empirical performance, aiming to chart a path forward and bridge this gap. We identify three critical properties—termed the \textbf{3D}-properties—that arise from DPO’s learning process: \textbf{D}rastic drop in the likelihood of rejected responses, \textbf{D}egradation into response suppression, and \textbf{D}ispersion effect on unseen responses. We show that these phenomena stem from the inherent features of DPO's optimization objective, where the interaction between the gradients of chosen and rejected responses causes instability. These findings are supported by experiments on both a carefully constructed toy model and practical LLM tasks, including mathematical problem-solving and instruction following. Our work offers new insights, connecting these observations to related research while providing a theoretical explanation for the underlying mechanisms. To address the challenges posed by the \textbf{3D}-properties, we propose straightforward regularization techniques that enhance training stability and final performance. Additionally, we investigate how the distribution of paired preference data affects DPO’s efficacy, contributing to a broader understanding of how alignment models handle out-of-domain (OOD) data. We believe our findings will help guide future research toward closing the gap between reward-model-free preference learning and reward-model-based approaches.

将大型语言模型 （LLMs人类偏好保持一致最近引起了广泛关注，其中近端策略优化 （PPO） 是一种规范但计算成本高昂的方法，而直接偏好优化 （DPO） 提供了一种更简单、更高效的替代方案。虽然之前的研究已经探讨了 PPO 和 DPO 之间的权衡，但 DPO 在最先进的生产级 LLMs，这表明存在潜在的局限性。在这项工作中，我们重新审视了 DPO，对其理论基础和实证表现进行了全面分析，旨在规划前进的道路并弥合这一差距。我们确定了 DPO 学习过程中产生的三个关键属性（称为 \textbf{3D} 属性）：被拒绝响应的可能性急剧下降，\textbf{D} 降级为响应抑制，以及 \textbf{D}ispersion 对看不见的响应的影响。我们表明，这些现象源于 DPO 优化目标的固有特征，其中选择和拒绝响应的梯度之间的相互作用会导致不稳定。这些发现得到了精心构建的玩具模型和实际 LLM 任务（包括数学问题解决和指令遵循）的实验的支持。我们的工作提供了新的见解，将这些观察结果与相关研究联系起来，同时为潜在机制提供了理论解释。为了解决 \textbf{3D} 属性带来的挑战，我们提出了简单的正则化技术，以提高训练稳定性和最终性能。此外，我们还研究了配对偏好数据的分布如何影响 DPO 的功效，有助于更广泛地了解比对模型如何处理域外 （OOD） 数据。 我们相信我们的发现将有助于指导未来的研究，以缩小无奖励模型的偏好学习和基于奖励模型的方法之间的差距。

### 主要内容
#### DPO三个关键属性(3D)
* **D**rastic drop in the likelihood of rejected responses被拒绝响应的可能性急剧下降
* **D**egradation into response suppression 降级为响应抑制
* **D**ispersion effect on unseen responses 对看不见的响应的影响

#### DPO不稳定的来源
* 选择和拒绝响应的梯度之间的相互作用会导致不稳定

#### 这个工作做了什么
* 联系观察结果和相关研究
* 为潜在机制提供理论解释
* 提供正则化技术解决3D属性带来的挑战
* 研究配对偏好数据的分布对DPO的影响

### 文章链接<a href='./papers/5344_3D_Properties_Identifying.pdf'>3D_Properties_Identifying</a>


## 2. Right Now, Wrong Then: Non-Stationary Direct Preference Optimization under Preference Drift
偏好漂移下的非平稳直接偏好优化

### 关键字
* LLM
* Fine-Tuning
* DPO
* non-stationarity 非平稳性
* preference drift 偏好偏移
* RLHF

### 摘要

Reinforcement learning from human feedback (RLHF) aligns Large Language Models (LLMs) with human preferences. However, these preferences can often change over time due to external factors (e.g. environment change and societal influence). Consequently, what was wrong then might be right now. Current preference optimization algorithms do not account for temporal preference drift in their modeling, which can lead to severe misalignment. To address this limitation, we use a Dynamic Bradley-Terry model that models preferences via time-dependent reward functions, and propose Non-Stationary Direct Preference Optimisation (NS-DPO). By introducing a discount parameter in the loss function, NS-DPO applies exponential weighting, which proportionally focuses learning on more time-relevant datapoints. We theoretically analyse the convergence of NS-DPO in the offline setting, providing upper bounds on the estimation error caused by non-stationary preferences. Finally, we demonstrate the effectiveness of NS-DPO1 for fine-tuning LLMs in scenarios with drifting preferences. By simulating preference drift using renowned reward models and modifying popular LLM datasets accordingly, we show that NS-DPO fine-tuned LLMs remain robust under non-stationarity, significantly outperforming baseline algorithms that ignore temporal preference changes, without sacrificing performance in stationary cases.
来自人类反馈的强化学习 （RLHF） 使大型语言模型 （LLMs人类偏好保持一致。然而，由于外部因素（例如环境变化和社会影响），这些偏好通常会随着时间的推移而改变。因此，当时的错误可能现在就是正确的。当前的偏好优化算法在其建模中没有考虑时间偏好漂移，这可能导致严重的错位。为了解决这一限制，我们使用了动态 Bradley-Terry 模型，该模型通过瞬态奖励函数对偏好进行建模，并提出了非平稳直接偏好优化 （NS-DPO）。通过在损失函数中引入 discount 参数，NS-DPO 应用指数加权，按比例将学习集中在与时间相关的更多数据点上。我们从理论上分析了 NS-DPO 在离线设置中的收敛性，提供了由非平稳偏好引起的估计误差的上限。最后，我们证明了 NS-DPO1 在具有漂移偏好的情况下微调 LLMs。通过使用著名的奖励模型模拟偏好漂移并相应地修改流行的 LLM 数据集，我们表明 NS-DPO 微调LLMs 在非平稳性下保持稳健性，明显优于忽略时间偏好变化的基线算法，而不会牺牲平稳情况下的性能。

### 主要内容
address the non-stationarity preference drift using exponential reweighting strategy(指数再加权策略) for LLMs.

### 文章链接<a href="./papers/3195_Right_Now_Wrong_Then_Non_.pdf">Non-Stationary Direct Preference Optimization under Preference Drift
</a>


s


## 占位
### 关键字
### 摘要
### 主要内容
### 文章链接



## 占位
### 关键字
### 摘要
### 主要内容
### 文章链接

## 占位
### 关键字
### 摘要
### 主要内容
### 文章链接




## 占位
### 关键字
### 摘要
### 主要内容
### 文章链接


## 占位
### 关键字
### 摘要
### 主要内容
### 文章链接


## 占位
### 关键字
### 摘要
### 主要内容
### 文章链接