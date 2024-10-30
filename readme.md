# ICLR2025中DPO(Direct Reference Optimization)相关文章收录

## 1. 3D-Properties: Identifying Challenges in DPO and Charting a Path Forward
3D 属性：识别 DPO 中的挑战并规划前进道路

### 关键字
* LLM (DPO主要用于大模型微调)
* DPO (直接偏好优化 Direct Preference Optimization)
* RLHF (Reinforcement Learning from Human Feedback)

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

### 文章链接
* <a href='./papers/5344_3D_Properties_Identifying.pdf'>查看PDF</a>
* <a href="https://openreview.net/forum?id=9Hxdixed7p">ICLR链接</a>

### 摘要
Aligning large language models (LLMs) with human preferences has recently garnered significant attention, with Proximal Policy Optimization (PPO) being a canonical yet computationally expensive method, and Direct Preference Optimization (DPO) offering a simpler and more efficient alternative. While prior studies have explored the trade-offs between PPO and DPO, DPO remains underutilized in state-of-the-art production-level LLMs, suggesting potential limitations. In this work, we revisit DPO with a comprehensive analysis of its theoretical foundations and empirical performance, aiming to chart a path forward and bridge this gap. We identify three critical properties—termed the \textbf{3D}-properties—that arise from DPO’s learning process: \textbf{D}rastic drop in the likelihood of rejected responses, \textbf{D}egradation into response suppression, and \textbf{D}ispersion effect on unseen responses. We show that these phenomena stem from the inherent features of DPO's optimization objective, where the interaction between the gradients of chosen and rejected responses causes instability. These findings are supported by experiments on both a carefully constructed toy model and practical LLM tasks, including mathematical problem-solving and instruction following. Our work offers new insights, connecting these observations to related research while providing a theoretical explanation for the underlying mechanisms. To address the challenges posed by the \textbf{3D}-properties, we propose straightforward regularization techniques that enhance training stability and final performance. Additionally, we investigate how the distribution of paired preference data affects DPO’s efficacy, contributing to a broader understanding of how alignment models handle out-of-domain (OOD) data. We believe our findings will help guide future research toward closing the gap between reward-model-free preference learning and reward-model-based approaches.
将大型语言模型 （LLMs人类偏好保持一致最近引起了广泛关注，其中近端策略优化 （PPO） 是一种规范但计算成本高昂的方法，而直接偏好优化 （DPO） 提供了一种更简单、更高效的替代方案。虽然之前的研究已经探讨了 PPO 和 DPO 之间的权衡，但 DPO 在最先进的生产级 LLMs，这表明存在潜在的局限性。在这项工作中，我们重新审视了 DPO，对其理论基础和实证表现进行了全面分析，旨在规划前进的道路并弥合这一差距。我们确定了 DPO 学习过程中产生的三个关键属性（称为 \textbf{3D} 属性）：被拒绝响应的可能性急剧下降，\textbf{D} 降级为响应抑制，以及 \textbf{D}ispersion 对看不见的响应的影响。我们表明，这些现象源于 DPO 优化目标的固有特征，其中选择和拒绝响应的梯度之间的相互作用会导致不稳定。这些发现得到了精心构建的玩具模型和实际 LLM 任务（包括数学问题解决和指令遵循）的实验的支持。我们的工作提供了新的见解，将这些观察结果与相关研究联系起来，同时为潜在机制提供了理论解释。为了解决 \textbf{3D} 属性带来的挑战，我们提出了简单的正则化技术，以提高训练稳定性和最终性能。此外，我们还研究了配对偏好数据的分布如何影响 DPO 的功效，有助于更广泛地了解比对模型如何处理域外 （OOD） 数据。 我们相信我们的发现将有助于指导未来的研究，以缩小无奖励模型的偏好学习和基于奖励模型的方法之间的差距。

## 2. Right Now, Wrong Then: Non-Stationary Direct Preference Optimization under Preference Drift
偏好漂移下的非平稳直接偏好优化

### 关键字
* LLM
* Fine-Tuning
* DPO
* non-stationarity 非平稳性
* preference drift 偏好偏移
* RLHF


### 主要内容

#### RLHF存在的问题
学习到的偏好可能会随时间推移而改变，于是当时的错误可能现在就是正确的：当前的偏好算法在建模中没有考虑时间偏好偏移
#### 如何解决
使用动态的Bradley-Terry模型，通过瞬时奖励函数对偏好进行建模，提出Non-Stationary DPO(非平稳直接偏好优化)
#### 技术细节
在损失函数中引入discount参数，NS-DPO应用指数加权，按比例学习集中在与时间相关的更多数据点上

一句话：
address the non-stationarity preference drift using exponential reweighting strategy(指数再加权策略) for LLMs.

### 文章链接
* <a href="./papers/3195_Right_Now_Wrong_Then_Non_.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=PabAln0jjB">ICLR中链接</a>

### 摘要
Reinforcement learning from human feedback (RLHF) aligns Large Language Models (LLMs) with human preferences. However, these preferences can often change over time due to external factors (e.g. environment change and societal influence). Consequently, what was wrong then might be right now. Current preference optimization algorithms do not account for temporal preference drift in their modeling, which can lead to severe misalignment. To address this limitation, we use a Dynamic Bradley-Terry model that models preferences via time-dependent reward functions, and propose Non-Stationary Direct Preference Optimisation (NS-DPO). By introducing a discount parameter in the loss function, NS-DPO applies exponential weighting, which proportionally focuses learning on more time-relevant datapoints. We theoretically analyse the convergence of NS-DPO in the offline setting, providing upper bounds on the estimation error caused by non-stationary preferences. Finally, we demonstrate the effectiveness of NS-DPO1 for fine-tuning LLMs in scenarios with drifting preferences. By simulating preference drift using renowned reward models and modifying popular LLM datasets accordingly, we show that NS-DPO fine-tuned LLMs remain robust under non-stationarity, significantly outperforming baseline algorithms that ignore temporal preference changes, without sacrificing performance in stationary cases.
来自人类反馈的强化学习 （RLHF） 使大型语言模型 （LLMs人类偏好保持一致。然而，由于外部因素（例如环境变化和社会影响），这些偏好通常会随着时间的推移而改变。因此，当时的错误可能现在就是正确的。当前的偏好优化算法在其建模中没有考虑时间偏好漂移，这可能导致严重的错位。为了解决这一限制，我们使用了动态 Bradley-Terry 模型，该模型通过瞬态奖励函数对偏好进行建模，并提出了非平稳直接偏好优化 （NS-DPO）。通过在损失函数中引入 discount 参数，NS-DPO 应用指数加权，按比例将学习集中在与时间相关的更多数据点上。我们从理论上分析了 NS-DPO 在离线设置中的收敛性，提供了由非平稳偏好引起的估计误差的上限。最后，我们证明了 NS-DPO1 在具有漂移偏好的情况下微调 LLMs。通过使用著名的奖励模型模拟偏好漂移并相应地修改流行的 LLM 数据集，我们表明 NS-DPO 微调LLMs 在非平稳性下保持稳健性，明显优于忽略时间偏好变化的基线算法，而不会牺牲平稳情况下的性能。

## 3. Iterative DPO with An Improvement Model for Fine-tuning Diffusion Models
针对微调扩散模型的具有改进模型的迭代DPO
### 关键字
* DPO
* Diffusion Models

### 主要内容
#### DPO问题
DPO可能收到离线偏好数据集的(offline preference dataset)约束
#### 解决措施
学习一个偏移改进模型，从偏好数据集中提取隐含偏好。然后使用学习到的改进模型从当前扩散模型生成的图像中生成获胜图像。通过使用当前扩散模型生成的图像作为失败图像，并将其相应的改进图像作为获胜图像来构建新的偏好数据对。
#### 作用效果
可以直接迭代应用在先偏好数据集来优化扩散模型，在离线DPO训练之外实现在线改进，无需额外人工标记或者冒着过度拟合奖励模型的风险
### 文章链接
* <a href="./papers/1368_Iterative_DPO_with_An_Imp.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=KJF3h0OpQ7">ICLR链接</a>

### 摘要
Direct Preference Optimization (DPO) has been proven as an effective solution in aligning generative models with human preferences. However, as shown in recent works, DPO could suffer from constraints from the offline preference dataset. This paper introduces a novel improvement approach for online iterative optimization of the diffusion models without introducing extra annotation of the online data. We propose to learn a preference improvement model to extract the implicit preference from the preference dataset. The learned improvement model is then used to generate winning images from the images generated by the current diffusion model. We can construct new pairs of preference data by using images generated by the current diffusion model as losing images, and its corresponding improved images as winning images. The diffusion model can therefore be optimized via iteratively applying online preference datasets. This method enables online improvement beyond offline DPO training without requiring additional human labeling or risking overfitting the reward model. Results demonstrate improvements in preference alignment with higher diversity compared with other fine-tuning methods. Our work bridges the gap between offline preference learning and online improvement, offering a promising direction for enhancing diffusion models in image generation tasks with limited preference data.
直接偏好优化 （DPO） 已被证明是使生成模型与人类偏好保持一致的有效解决方案。然而，正如最近的工作所示，DPO 可能会受到离线偏好数据集的约束。本文介绍了一种新的改进方法，用于扩散模型的在线迭代优化，而无需对在线数据进行额外的注释。我们建议学习一个偏好改进模型，从偏好数据集中提取隐含偏好。然后使用学习到的改进模型从当前扩散模型生成的图像中生成获胜图像。我们可以通过使用当前扩散模型生成的图像作为失败图像，并将其相应的改进图像作为获胜图像来构建新的偏好数据对。因此，可以通过迭代应用在线偏好数据集来优化扩散模型。这种方法可以在离线 DPO 训练之外实现在线改进，而无需额外的人工标记或冒着过度拟合奖励模型的风险。结果表明，与其他微调方法相比，更高的多样性在偏好对齐方面有所改善。我们的工作弥合了离线偏好学习和在线改进之间的差距，为在偏好数据有限的图像生成任务中增强扩散模型提供了一个有前途的方向。

## 4. Provably Mitigating Corruption, Overoptimization, and Verbosity Simultaneously in Offline and Online RLHF/DPO Alignment
在离线和在线RLHF/DPO对齐中**同时缓解损坏，过度优化和详细程度**
### 关键字
* RLHF
* DPO
* Large Language Models
* Alignment 
### 主要内容
#### RLHF和DPO训练过程的问题
RLHF和DPO训练质量由于
* Corrupted Preference 偏好受限
* Reward Overoptimization 奖励过度优化
* Bias towards Verbosity 偏重口头禅
而严重受限
#### 其他文章存在的问题
大多数文章直解决其中的一个，而其他少数工作需要大量的计算来估计多个奖励模型，缺乏泛化能力的理论保证
#### 提出RLHF-COV和DPO-COV
COV是指:(Corruption, Overoptimization & Verbosity)
#### 作用效果
可以在在线和离线学习中同时缓解这三个问题(COV)

这种能力可以通过为在损坏数据上训练的 DPO-COV 算法获得长度正则化泛化错误率来理论上证明，这与具有干净数据且没有长度正则化的简单情况的最已知率相匹配。

DPO-COV无需奖励估计即可实现
### 文章链接
* <a href="./papers/11600_Provably_Mitigating_Corr.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=K2OWrXUVby">ICLR链接</a>

### 摘要
Reinforcement learning from human feedback (RLHF) and direct preference optimization (DPO) are emerging and important techniques to align large language models (LLM) with human preference. However, the quality of RLHF and DPO training is seriously compromised by Corrupted preference, reward Overoptimization, and bias towards Verbosity. To our knowledge, most existing works tackle only one of these important issues, and the few other works require much computation to estimate multiple reward models and lack theoretical guarantee of generalization ability. In this work, we propose RLHF-COV and DPO-COV algorithms that can simultaneously mitigate these three issues, in both offline and online settings. This ability is theoretically demonstrated by obtaining length-regularized generalization error rates for our DPO-COV algorithms trained on corrupted data, which match the best-known rates for simpler cases with clean data and without length regularization. Moreover, our DPO-COV algorithm is simple to implement without reward estimation, and is proved to be equivalent to our RLHF-COV algorithm, which directly implies the equivalence between the vanilla RLHF and DPO algorithms.
来自人类反馈的强化学习 （RLHF） 和直接偏好优化 （DPO） 是新兴的重要技术，用于将大型语言模型 （LLM人类偏好保持一致。然而，RLHF 和 DPO 训练的质量受到 C或偏好、奖励 O验证和对 Verbosity 的偏向的严重影响。据我们所知，现有的大多数工作只解决了这些重要问题中的一个，而其他少数工作需要大量的计算来估计多个奖励模型，缺乏泛化能力的理论保证。在这项工作中，我们提出了 RLHF-COV 和 DPO-COV 算法，它们可以在离线和在线环境中同时缓解这三个问题。这种能力可以通过为在损坏数据上训练的 DPO-COV 算法获得长度正则化泛化错误率来理论上证明，这与具有干净数据且没有长度正则化的简单情况的最已知率相匹配。此外，我们的 DPO-COV 算法无需奖励估计即可轻松实现，并被证明等同于我们的 RLHF-COV 算法，这直接暗示了原版 RLHF 和 DPO 算法之间的等效性。

## 5. UNA: Unifying Alignments of RLHF/PPO, DPO and KTO by a Generalized Implicit Reward Function
通过广义隐式奖励函数统一 RLHF/PPO、DPO 和 KTO 的对齐
### 关键字
* LLM Alignment
* Unified Alignment
* RLHF, PPO, DPO, KTO(Knowledge Transfer Optimization)
### 主要内容
#### 各种alignment技术的局限性
* RLHF需要同时分别训练奖励模型和策略，训练过程复杂、耗时、内存密集且不稳定
* DPO提出最优策略和奖励之间的映射从而简化RLHF的训练过程，但是不能充分利用模型，且仅限于成对偏好数据
#### 提出UNA--Unified Alignment
* 数学上证明对于经典的RLHF目标，最优策略是由广义隐式奖励函数诱导的
* 之后说明UNA的作用
#### UNA效果
1. 将RLHF/PPO, DPo, KTO统一为一种监督学习，以最小化隐形奖励(implicit reward)和显示奖励(explicit reward)之间的差异
2. 优于RLHF/PPO，同时简化、稳定、加速和减轻RL微调过程的内存负担
3. 适用不同的反馈类型，包括成对、二进制和标量反馈

### 文章链接
<a href="./papers/8566_UNA_Unifying_Alignments_o.pdf">查看PDF</a>
<a href="https://openreview.net/forum?id=ZSbsX1sFo3">ICLR链接</a>

### 摘要
An LLM is pretrained on trillions of tokens, but the pretrained LLM may still generate undesired responses. To solve this problem, alignment techniques such as RLHF, DPO and KTO are proposed. However, these alignment techniques have limitations. For example, RLHF requires training the reward model and policy separately, which is complex, time-consuming, memory intensive and unstable during training processes. DPO proposes a mapping between an optimal policy and a reward, greatly simplifying the training process of RLHF. However, it can not take full advantages of a reward model and it is limited to pairwise preference data.
In this paper, we propose \textbf{UN}ified \textbf{A}lignment (UNA) which unifies RLHF/PPO, DPO and KTO. Firstly, we mathematically prove that given the classical RLHF objective, the optimal policy is induced by a generalize implicit reward function. With this novel mapping between a reward model and an optimal policy, UNA can 1. unify RLHF/PPO, DPO and KTO into a supervised learning of minimizing the difference between an implicit reward and an explicit reward; 2. outperform RLHF/PPO while simplify, stabilize, speed up and reduce memory burden of RL fine-tuning process; 3. accommodate different feedback types including pairwise, binary and scalar feedback. Downstream experiments show UNA outperforms DPO, KTO and RLHF.
LLM 在数万亿个令牌上进行了预训练，但预训练的 LLM 可能仍会生成不需要的响应。为了解决这个问题，提出了 RLHF 、 DPO 和 KTO 等对准技术。但是，这些对齐技术具有局限性。例如，RLHF 需要分别训练奖励模型和策略，训练过程中复杂、耗时、内存密集且不稳定。DPO 提出了最优策略和奖励之间的映射，大大简化了 RLHF 的训练过程。但是，它不能充分利用奖励模型，并且仅限于成对偏好数据。
在本文中，我们提出了 \textbf{UN}ified \textbf{A}lignment （UNA），它统一了 RLHF/PPO、DPO 和 KTO。首先，我们从数学上证明，给定经典的 RLHF 目标，最优策略是由广义隐式奖励函数诱导的。通过奖励模型和最佳政策之间的这种新颖映射，UNA 可以 1.将 RLHF/PPO、DPO 和 KTO 统一为一种监督学习，以最小化隐性奖励和显式奖励之间的差异;2. 优于 RLHF/PPO，同时简化、稳定、加速和减轻 RL 微调过程的内存负担;3. 适应不同的反馈类型，包括成对、二进制和标量反馈。下游实验表明 UNA 优于 DPO 、 KTO 和 RLHF。



## 6. The Crucial Role of Samplers in Online Direct Preference Optimization
采样器(Sampler)在在线DPO中的关键作用
### 关键字
* DPO
* multi-armed bandit多臂老虎机(MAB问题是一个exploration vs. exploitation优化问题)

### 主要内容
#### 探索采样器对DPO收敛率的影响
#### 探索并对比不同采样策略
* 均匀采样实现线性收敛
* 文章提出的**在线采样器**实现二次收敛
* 结合后验分布和Logit混合进一步使采样器使用实际设置，取得很大改进
    * logit混合是一种集成方法，通过将多个模型的输出进行加权组合以生成更加稳定和准确的预测
### 文章链接
* <a href="./papers/1735_The_Crucial_Role_of_Sampl.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=F6z3utfcYw">ICLR链接</a>
### 摘要
Direct Preference Optimization (DPO) has emerged as a stable, scalable, and efficient solution for language model alignment. Despite its empirical success, the optimization properties, particularly the impact of samplers on its convergence rates, remain underexplored. In this paper, we provide a rigorous analysis of DPO's convergence rates with different sampling strategies under the exact gradient setting, revealing a surprising separation: uniform sampling achieves linear convergence, while our proposed online sampler achieves quadratic convergence. We further adapt the sampler to practical settings by incorporating posterior distributions and logit mixing, demonstrating significant improvements over previous approaches. On Safe-RLHF dataset, our method exhibits a % improvement over vanilla DPO and a % improvement over on-policy DPO; on Iterative-Prompt, our approach outperforms vanilla DPO, on-policy DPO, and Hybrid GSHF by over %. Our results not only offer insights into the theoretical standing of DPO but also pave the way for potential algorithm designs in the future.
直接偏好优化 （DPO） 已成为一种稳定、可扩展且高效的语言模型对齐解决方案。 尽管它在实证上取得了成功，但优化特性，特别是采样器对其收敛率的影响，仍未得到充分探索。在本文中，我们对在精确梯度设置下采用不同采样策略的 DPO 收敛率进行了严格的分析，揭示了一个令人惊讶的分离：均匀采样实现线性收敛，而我们提出的在线采样器实现二次收敛。我们通过结合后验分布和 logit 混合进一步使采样器适应实际设置，展示了与以前方法相比的显着改进。在 Safe-RLHF 数据集上，我们的方法比普通 DPO 提高了 4.5%，比政策 DPO 提高了 3.0%; 在迭代提示上，我们的方法比原版 DPO、策略 DPO 和混合 GSHF 高出 4.2% 以上 。我们的结果不仅为DPO的理论地位提供了见解，也为未来潜在的算法设计铺平了道路。

## 7. Towards Robust Alignment of Language Models: Distributionally Robustifying Direct Preference Optimization
向语言模型的鲁棒对齐：分布稳健的DPO
### 关键字
* DPO
* LLM Alignment
* Distributionally Robust Optimization
### 主要内容
#### 针对DPO训练数据集中的噪声挑战
LLM中的噪声：
* 逐点噪声(pointwise noise)：包括低质量数据点
* 成对噪声(pairwise noise)：影响偏好排名的错误数据对关联
#### 
### 文章链接
* <a href="./papers/10293_Towards_Robust_Alignment.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=CbfsKHiWEn">ICLR链接</a>
### 摘要
This study addresses the challenge of noise in training datasets for Direct Preference Optimization (DPO), a method for aligning Large Language Models (LLMs) with human preferences. We categorize noise into pointwise noise, which includes low-quality data points, and pairwise noise, which encompasses erroneous data pair associations that affect preference rankings. Utilizing Distributionally Robust Optimization (DRO), we enhance DPO's resilience to these types of noise. Our theoretical insights reveal that DPO inherently embeds DRO principles, conferring robustness to pointwise noise, with the regularization coefficient playing a critical role in its noise resistance. Extending this framework, we introduce Distributionally Robustifying DPO (Dr. DPO), which integrates pairwise robustness by optimizing against worst-case pairwise scenarios. The novel hyperparameter in Dr. DPO allows for fine-tuned control over data pair reliability, providing a strategic balance between exploration and exploitation in noisy training environments. Empirical evaluations demonstrate that Dr. DPO substantially improves the quality of generated text and response accuracy in preference datasets, showcasing enhanced performance in both noisy and noise-free settings.
本研究解决了直接偏好优化 （DPO） 训练数据集中的噪声挑战，DPO 是一种将大型语言模型 （LLMs。我们将噪声分为逐点噪声（包括低质量数据点）和成对噪声（包含影响偏好排名的错误数据对关联）。利用分布稳健优化 （DRO），我们增强了 DPO 对这些类型噪声的弹性。我们的理论见解表明，DPO 本身嵌入了 DRO 原理，赋予了逐点噪声的鲁棒性，正则化系数在其抗噪声性 中起着关键作用。扩展此框架，我们引入了分布稳健性 DPO （Dr. DPO），它通过针对最坏情况的成对情景进行优化来集成成对稳健性。Dr. DPO 中的新型超参数 允许对数据对可靠性进行微调控制，在嘈杂的训练环境中提供勘探和开发之间的战略平衡。实证评估表明，DPO 博士大大提高了偏好数据集中生成文本的质量和响应准确性，在有噪声和无噪声设置中都表现出增强的性能。
## 占位
### 关键字
### 主要内容
### 文章链接
### 摘要

## 占位
### 关键字
### 主要内容
### 文章链接
### 摘要

## 占位
### 关键字
### 主要内容
### 文章链接
### 摘要

## 占位
### 关键字
### 主要内容
### 文章链接
### 摘要

## 占位
### 关键字
### 主要内容
### 文章链接
### 摘要