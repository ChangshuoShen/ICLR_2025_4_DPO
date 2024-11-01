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
#### 使用DRO优化DPO: Dr. DPO
* 利用分布稳健模型(Distributionally Robust Optimization)来增强DPO对上述噪声的弹性
* 理论上说明DPO本身潜入了DRO原理，赋予了其对噪声的鲁棒性，正则化系数在其抗噪性$\beta$中起关键作用
* 拓展引入Dr.DPO (distributionally robust DPO)，通过针对最坏情况的成对场景来集成成对稳健性

### 文章链接
* <a href="./papers/10293_Towards_Robust_Alignment.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=CbfsKHiWEn">ICLR链接</a>
### 摘要
This study addresses the challenge of noise in training datasets for Direct Preference Optimization (DPO), a method for aligning Large Language Models (LLMs) with human preferences. We categorize noise into pointwise noise, which includes low-quality data points, and pairwise noise, which encompasses erroneous data pair associations that affect preference rankings. Utilizing Distributionally Robust Optimization (DRO), we enhance DPO's resilience to these types of noise. Our theoretical insights reveal that DPO inherently embeds DRO principles, conferring robustness to pointwise noise, with the regularization coefficient playing a critical role in its noise resistance. Extending this framework, we introduce Distributionally Robustifying DPO (Dr. DPO), which integrates pairwise robustness by optimizing against worst-case pairwise scenarios. The novel hyperparameter in Dr. DPO allows for fine-tuned control over data pair reliability, providing a strategic balance between exploration and exploitation in noisy training environments. Empirical evaluations demonstrate that Dr. DPO substantially improves the quality of generated text and response accuracy in preference datasets, showcasing enhanced performance in both noisy and noise-free settings.
本研究解决了直接偏好优化 （DPO） 训练数据集中的噪声挑战，DPO 是一种将大型语言模型 （LLMs。我们将噪声分为逐点噪声（包括低质量数据点）和成对噪声（包含影响偏好排名的错误数据对关联）。利用分布稳健优化 （DRO），我们增强了 DPO 对这些类型噪声的弹性。我们的理论见解表明，DPO 本身嵌入了 DRO 原理，赋予了逐点噪声的鲁棒性，正则化系数在其抗噪声性 中起着关键作用。扩展此框架，我们引入了分布稳健性 DPO （Dr. DPO），它通过针对最坏情况的成对情景进行优化来集成成对稳健性。Dr. DPO 中的新型超参数 允许对数据对可靠性进行微调控制，在嘈杂的训练环境中提供勘探和开发之间的战略平衡。实证评估表明，DPO 博士大大提高了偏好数据集中生成文本的质量和响应准确性，在有噪声和无噪声设置中都表现出增强的性能。


## 8. Length Desensitization in Direct Preference Optimization
DPO中的长度脱敏
### 关键字
* LLM
* RLHF
* PO(Preference Optimization)
### 主要内容
#### DPO对详细程度有化过度
往往会针对详细程度过度优化，可能会对性能和用户体验产生不利影响
#### 分析得“长度敏感性”
通过对DPO的优化目标做理论分析，得到其隐含奖励与数据长度之间的强相关性，会误导优化方向，导致DPO的长度敏感并倾向于冗长
#### LD(length desensitization)-DPO
所提出的方法旨在通过将相对不重要的显式长度偏好与其他隐式偏好解耦，使 DPO 对数据长度脱敏，从而能够更有效地学习内在偏好。

### 文章链接
* <a href="./papers/6752_Length_Desensitization_in.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=CuwjD3cazX">ICLR链接</a>
### 摘要
Direct Preference Optimization (DPO) is widely utilized in the Reinforcement Learning from Human Feedback (RLHF) phase to align Large Language Models (LLMs) with human preferences, thereby enhancing both their harmlessness and efficacy. However, it has been observed that DPO tends to over-optimize for verbosity, which can detrimentally affect both performance and user experience. In this paper, we conduct an in-depth theoretical analysis of DPO's optimization objective and reveal a strong correlation between its implicit reward and data length. This correlation misguides the optimization direction, resulting in length sensitivity during the DPO training and leading to verbosity. To address this issue, we propose a length-desensitization improvement method for DPO, termed LD-DPO. The proposed method aims to desensitize DPO to data length by decoupling explicit length preference, which is relatively insignificant, from the other implicit preferences, thereby enabling more effective learning of the intrinsic preferences. We utilized two settings (Base and Instruct) of Llama2-13B, Llama3-8B, and Qwen2-7B for experimental validation on various benchmarks including MT-Bench and AlpacaEval 2. The experimental results indicate that LD-DPO consistently outperforms DPO and other baseline methods, achieving more concise responses with a 10-40% reduction in length compared to DPO. We conducted in-depth experimental analyses to demonstrate that LD-DPO can indeed achieve length desensitization and align the model more closely with human-like preferences. ”Brevity is the Soul of Wit.''—William Shakespeare
直接偏好优化 （DPO） 广泛用于人类反馈强化学习 （RLHF） 阶段，以使大型语言模型 （LLMs人类偏好保持一致，从而提高它们的无害性和有效性。但是，据观察，DPO 往往会针对详细程度进行过度优化，这可能会对性能和用户体验产生不利影响。在本文中，我们对 DPO 的优化目标进行了深入的理论分析，并揭示了其隐含奖励与数据长度之间的强相关性。这种相关性误导了优化方向，导致 DPO 训练期间的长度敏感并导致冗长。为了解决这个问题，我们提出了一种 DPO 的长度脱敏改进方法，称为 LD-DPO。所提出的方法旨在通过将相对微不足道的显式长度偏好与其他隐式偏好解耦，使 DPO 对数据长度脱敏，从而能够更有效地学习内在偏好。我们使用了 Llama2-13B、Llama3-8B 和 Qwen2-7B 的两种设置 （Base 和 Instruct） 在各种基准（包括 MT-Bench 和 AlpacaEval 2）上进行实验验证。实验结果表明，LD-DPO 始终优于 DPO 和其他基线方法，与 DPO 相比，实现了更简洁的反应，长度减少了 10-40%。我们进行了深入的实验分析，以证明 LD-DPO 确实可以实现长度脱敏，并使模型更紧密地与类似人类的偏好保持一致。“简洁是机智的灵魂。”——威廉·莎士比亚

## 9. Bootstrapping Language Models with DPO Implicit Rewards
使用 DPO 隐式奖励引导语言模型
### 关键字
* Alignment
* DPO
* LLM

### 主要内容
#### 充分利用DPO训练后得到的隐式奖励模型
即这种隐式奖励模型本身可以以引导方式使用，以进一步对齐 LLM
#### 细节
使用当前 LLM来构建偏好数据集，然后在后续的 DPO 轮次中使用。合并消除响应长度偏差并提高偏好数据集质量的改进，以进一步提高。
#### DICE(Dpo ImpliCit rEwards) self-alignment
### 文章链接
* <a href="./papers/6460_Bootstrapping_Language_Mo.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=dliIIodM6b">ICLR连接</a>
### 摘要
Human alignment in large language models (LLMs) is an active area of research. A recent groundbreaking work, direct preference optimization (DPO), has greatly simplified the process from past work in reinforcement learning from human feedback (RLHF) by bypassing the reward learning stage in RLHF. DPO, after training, provides an implicit reward model. In this work, we make a novel observation that this implicit reward model can by itself be used in a bootstrapping fashion to further align the LLM. Our approach is to use the rewards from a current LLM model to construct a preference dataset, which is then used in subsequent DPO rounds. We incorporate refinements that debias the length of the responses and enhance the quality of the preference dataset to further improve our approach. Our approach, named self-alignment with DPO ImpliCit rEwards (DICE), shows great improvements in alignment. It achieves an increase of more than 8 in length-controlled win rate on AlpacaEval 2 for all the different base models that we tried, without relying on external feedback.
大型语言模型中的人类对齐 （LLMs） 是一个活跃的研究领域。最近的一项开创性工作，直接偏好优化 （DPO），通过绕过 RLHF 中的奖励学习阶段，大大简化了过去从人类反馈强化学习 （RLHF） 的工作过程。DPO 在训练后提供了一个隐式奖励模型。在这项工作中，我们提出了一个新颖的观察，即这种隐式奖励模型本身可以以引导方式使用，以进一步对齐 LLM。我们的方法是使用当前 LLM来构建偏好数据集，然后在后续的 DPO 轮次中使用。我们合并了消除响应长度偏差并提高偏好数据集质量的改进，以进一步改进我们的方法。我们的方法被命名为 DPO ImpliCit rEwards （DICE） 的自对准，显示出对准的巨大改进。对于我们尝试过的所有不同基本模型，它在 AlpacaEval 2 上实现了超过 8 的长度控制胜率增加，而无需依赖外部反馈。

## 10. Effective Text-to-Image Alignment with Quality Aware Pair Ranking
有效的文生图在质量感知排名的对齐
### 关键字
* DPO
* Diffusion
### 主要内容
#### Diffusion-DPO在T2I(Text to Image)领域有效
#### 提出问题
所有首选项对对齐微调的贡献是否相同？偏好有时可能是主观的，并且可能并不总是转化为有效地调整模型
#### 工作重点——开发一个质量指标
开发了一个质量指标来对图像偏好对进行排名，并实现有效的基于 Diffusion-DPO 的对齐微调。

### 文章链接
* <a href="./papers/8100_Effective_Text_to_Image_A.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=YeZNN6Iy6Q">ICLR链接</a>
### 摘要
Fine-tuning techniques such as Reinforcement Learning with Human Feedback (RLHF) and Direct Preference Optimization (DPO) allow us to steer Large Language Models (LLMs) to be align better with human preferences. Alignment is equally important in text-to-image generation. Recent adoption of DPO, specifically Diffusion-DPO, for Text-to-Image (T2I) diffusion models has proven to work effectively in improving visual appeal and prompt-image alignment. The mentioned works fine-tune on Pick-a-Pic dataset, consisting of approximately one million image preference pairs, collected via crowdsourcing at scale. However, do all preference pairs contribute equally to alignment fine-tuning? Preferences can be subjective at times and may not always translate into effectively aligning the model. In this work, we investigate the above-mentioned question. We develop a quality metric to rank image preference pairs and achieve effective Diffusion-DPO-based alignment fine-tuning.We show that the SD-1.5 and SDXL models fine-tuned using the top 5.33% of the data perform better both quantitatively and qualitatively than the models fine-tuned on the full dataset.
基于人工反馈的强化学习 （RLHF） 和直接偏好优化 （DPO） 等微调技术使我们能够引导大型语言模型 （LLMs） 更好地与人类偏好保持一致。对齐方式在文本到图像的生成中同样重要。最近将 DPO，特别是 Diffusion-DPO 用于文本到图像 （T2I） 扩散模型已被证明可以有效地提高视觉吸引力和提示图像对齐。上述工作在 Pick-a-Pic 数据集上进行微调，该数据集由大约 100 万个图像偏好对组成，通过大规模众包收集。但是，所有首选项对对齐微调的贡献是否相同？偏好有时可能是主观的，并且可能并不总是转化为有效地调整模型。在这项工作中，我们研究了上述问题。我们开发了一个质量指标来对图像偏好对进行排名，并实现有效的基于 Diffusion-DPO 的对齐微调。我们表明，使用前 5.33% 的数据进行微调的 SD-1.5 和 SDXL 模型在定量和定性方面都比在整个数据集上微调的模型表现更好。


## 11. alpha-DPO: Adaptive Reward Margin is What Direct Preference Optimization Needs
DPO需要自适应奖励边际
### 关键字
* DPO
* LLM'a Alignment
### 主要内容
#### 现有问题
* RLHF的计算效率和训练稳定性不好
* DPO和SimPO提供RLHF的离线替代方案，但是
    * DPO依赖潜在的次优参考模型(potentially suboptimal reference model)
    * SimPO对固定目标奖励边际的假设可能会导致在不同数据设置中做出次优决策
#### 他们提出了alpha-DPO
* 自适应偏好优化算法(adaptive preference optimization algorithm)，通过引入动态奖励边际来解决上述限制
* 使用自适应偏好分配，平衡策略模型与参考模型，以实现个性化的奖励边际
* 理论证明了它作为替代优化目标的有效性以及它通过 KL 发散控制平衡对齐和多样性的能力。
### 文章链接
* <a href="./papers/13547_alpha_DPO_Adaptive_Rewar.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=QqziJAdev9">ICLR链接</a>
### 摘要
Aligning large language models (LLMs) with human values and intentions is crucial for their utility, honesty, and safety. Reinforcement learning from human feedback (RLHF) is a popular approach to achieve this alignment, but it faces challenges in computational efficiency and training stability. Recent methods like Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO) have proposed offline alternatives to RLHF, simplifying the process by reparameterizing the reward function. However, DPO depends on a potentially suboptimal reference model, and SimPO's assumption of a fixed target reward margin may lead to suboptimal decisions in diverse data settings. In this work, we propose (\alpha)-DPO, an adaptive preference optimization algorithm designed to address these limitations by introducing a dynamic reward margin. Specifically, (\alpha)-DPO employs an adaptive preference distribution, balancing the policy model and the reference model to achieve personalized reward margins. We provide theoretical guarantees for (\alpha)-DPO, demonstrating its effectiveness as a surrogate optimization objective and its ability to balance alignment and diversity through KL divergence control. Empirical evaluations on AlpacaEval 2 and Arena-Hard show that (\alpha)-DPO consistently outperforms DPO and SimPO across various model settings, establishing it as a robust approach for fine-tuning LLMs. Our method achieves significant improvements in win rates, highlighting its potential as a powerful tool for LLM alignment.a
将大型语言模型 （LLMs人类价值观和意图保持一致，对于它们的实用性、诚实性和安全性至关重要。RLHF 是实现这种对齐的常用方法，但它在计算效率和训练稳定性方面面临挑战。最近的方法，如直接偏好优化 （DPO） 和简单偏好优化 （SimPO） 提出了 RLHF 的离线替代方案，通过重新参数化奖励函数来简化流程。然而，DPO 依赖于潜在的次优参考模型，而 SimPO 对固定目标奖励边际的假设可能会导致在不同数据设置中做出次优决策。在这项工作中，我们提出了 （\alpha）-DPO，这是一种自适应偏好优化算法，旨在通过引入动态奖励边际来解决这些限制。具体来说， （\alpha）-DPO 采用自适应偏好分配，平衡策略模型和参考模型，以实现个性化的奖励边际。我们为 （\alpha）-DPO 提供了理论保证，证明了它作为替代优化目标的有效性以及它通过 KL 发散控制平衡对齐和多样性的能力。对 AlpacaEval 2 和 Arena-Hard 的实证评估表明，（\alpha）-DPO 在各种模型设置中始终优于 DPO 和 SimPO，使其成为微调 LLMs。我们的方法在胜率方面取得了显着的提高，凸显了它作为 LLM。

## 12. Mask-DPO: Generalizable Fine-grained Factuality Alignment of LLMs
通用细粒度事实性对齐
### 关键字
* Hallucination Mitigation 幻觉缓解（通常涉及引导模型聚焦于真实、准确的信息，使得AI助手更可信。）
* Large Language Model
* Fine-grained Alignment 细粒度对齐（通过更精确的方式进行模型训练，使其输出内容更加真实、准确。在这个上下文中，它意味着模型可以对每个句子级别的事实性进行对齐，而不仅仅是整体响应，确保只学习真实句子的内容。）

### 主要内容
#### 面对的问题
* LLM在各个领域充当AI助手时会出现幻觉
* 以前进行响应水平偏好学习的事实对齐方法在训练过程中不可避免地引入了噪音
#### Mask-DPO技术细节
将句子级事实性作为mask signal，只从首选样本中事实正确的句子中学习，并防止非首选样本中的事实内容受到惩罚，从而解决了偏好学习中的歧义。
#### 额外结论：
使用不同的训练样本缩放策略进一步研究 Mask-DPO 的泛化特性，发现扩展数据集中的主题数量比增加问题数量更有效。
### 文章链接
* <a href="./papers/4078_Mask_DPO_Generalizable_Fi.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=d2H1oTNITn">ICLR链接</a>
### 摘要
Large language models (LLMs) exhibit hallucinations (i.e., unfaithful or nonsensical information) when serving as AI assistants in various domains. Since hallucinations always come with truthful content in the LLM responses, previous factuality alignment methods that conduct response-level preference learning inevitably introduced noises during training. Therefore, this paper proposes a fine-grained factuality alignment method based on Direct Preference Optimization (DPO), called Mask-DPO. Incorporating sentence-level factuality as mask signals, Mask-DPO only learns from factually correct sentences in the preferred samples and prevents the penalty on factual contents in the not preferred samples, which resolves the ambiguity in the preference learning. Extensive experimental results demonstrate that Mask-DPO can significantly improve the factuality of LLMs responses to questions from both in-domain and out-of-domain datasets, although these questions and their corresponding topics are unseen during training. Only trained on the ANAH train set, the score of Llama3.1-8B-Instruct on the ANAH test set is improved from 49.19% to 77.53%, even surpassing the score of Llama3.1-70B-Instruct (53.44%), while its FactScore on the out-of-domain Biography dataset is also improved from 30.29% to 39.39%. We further study the generalization property of Mask-DPO using different training sample scaling strategies and find that scaling the number of topics in the dataset is more effective than the number of questions. We provide a hypothesis of what factual alignment is doing with LLMs, on the implication of this phenomenon, and conduct proof-of-concept experiments to verify it. We hope the method and the findings pave the way for future research on scaling factuality alignment.
大型语言模型 （LLMs） 在各个领域充当 AI 助手时会出现幻觉（即不忠实或荒谬的信息）。由于幻觉总是伴随着 LLM，因此以前进行响应水平偏好学习的事实对齐方法在训练过程中不可避免地引入了噪音。因此，本文提出了一种基于直接偏好优化 （DPO） 的细粒度事实对齐方法，称为 Mask-DPO。Mask-DPO 将句子级事实性作为掩码信号，只从首选样本中事实正确的句子中学习，并防止非首选样本中的事实内容受到惩罚，从而解决了偏好学习中的歧义。广泛的实验结果表明，Mask-DPO 可以显著提高 LLMs 对域内和域外数据集问题回答的真实性，尽管这些问题及其相应的主题在训练期间是看不到的。仅在 ANAH 训练集上训练，Llama3.1-8B-Instruct 在 ANAH 测试集上的分数从 49.19% 提高到 77.53%，甚至超过了 Llama3.1-70B-Instruct 的分数（53.44%），而它在域外传记数据集上的 FactScore 也从 30.29% 提高到 39.39%。我们使用不同的训练样本缩放策略进一步研究了 Mask-DPO 的泛化特性，发现扩展数据集中的主题数量比增加问题数量更有效。我们提供了一个假设，即事实对齐对 LLMs 的影响，关于这种现象的含义，并进行了概念验证实验来验证它。我们希望该方法和发现为未来关于缩放事实对齐的研究铺平道路。


## 13. Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs
长链推理的逐步偏好优化
### 关键字
* LLM
* Mathematical Reasoning
* DPO
### 主要内容
#### 问题
* 数学推导需要广泛而精确的推理链，对LLM挑战巨大
* DPO对长链数学推理好处有限，因为采用 DPO 的模型难以识别错误答案中的详细错误。
* 文中说：这种限制源于缺乏精细的过程监督。
#### 提出的解决措施Step-DPO
将**单个推理步骤**视为偏好优化的单元，而不是整体评估答案
#### 附加的工作
* 为 Step-DPO 开发了一个数据构建管道，能够创建包含 10K 逐步偏好对的高质量数据集
* 发现：在 DPO 中，政策模型生成的数据比人类或 GPT-4 生成的数据更有效
### 文章链接
<a href="./papers/1691_Step_DPO_Step_wise_Prefer.pdf">查看PDF</a>
<a href="https://openreview.net/forum?id=H5FUVj0vMd">ICLR链接</a>

### 摘要
Mathematical reasoning presents a significant challenge for Large Language Models (LLMs) due to the extensive and precise chain of reasoning required for accuracy. Ensuring the correctness of each reasoning step is critical. To address this, we aim to enhance the robustness and factuality of LLMs by learning from human feedback. However, Direct Preference Optimization (DPO) has shown limited benefits for long-chain mathematical reasoning, as models employing DPO struggle to identify detailed errors in incorrect answers. This limitation stems from a lack of fine-grained process supervision. We propose a simple, effective, and data-efficient method called Step-DPO, which treats individual reasoning steps as units for preference optimization rather than evaluating answers holistically. Additionally, we have developed a data construction pipeline for Step-DPO, enabling the creation of a high-quality dataset containing 10K step-wise preference pairs. We also observe that in DPO, the data generated by the policy model is more effective than that produced by humans or GPT-4, due to the former's in-distribution nature. Our findings demonstrate that as few as 10K preference data pairs and fewer than 500 Step-DPO training steps can yield a nearly 3% gain in accuracy on MATH for models with over 70B parameters. Notably, Step-DPO, when applied to Qwen2-72B-Instruct, achieves scores of 70.8% and 94.0% on the test sets of MATH and GSM8K, respectively, surpassing a series of closed-source models, including GPT-4-1106, Claude-3-Opus, and Gemini-1.5-Pro.
数学推理对大型语言模型 （LLMs，因为准确性需要广泛而精确的推理链。确保每个推理步骤的正确性至关重要。为了解决这个问题，我们的目标是通过学习人类反馈来提高 LLMs。然而，直接偏好优化 （DPO） 对长链数学推理的好处有限，因为采用 DPO 的模型难以识别错误答案中的详细错误。这种限制源于缺乏精细的过程监督。我们提出了一种简单、有效且数据高效的方法，称为 Step-DPO，它将单个推理步骤视为偏好优化的单元，而不是整体评估答案。此外，我们还为 Step-DPO 开发了一个数据构建管道，能够创建包含 10K 逐步偏好对的高质量数据集。我们还观察到，在 DPO 中，由于前者的分布性质，政策模型生成的数据比人类或 GPT-4 生成的数据更有效。我们的研究结果表明，对于参数超过 70B 的模型，只要 10K 个偏好数据对和少于 500 个 Step-DPO 训练步骤，就可以使 MATH 的准确性提高近 3%。值得注意的是，当 Step-DPO 应用于 Qwen2-72B-Struct 时，在 MATH 和 GSM8K 的测试集上分别取得了 70.8% 和 94.0% 的分数，超过了包括 GPT-4-1106、Claude-3-Opus 和 Gemini-1.5-Pro 在内的一系列闭源模型。

## 14. TIS-DPO: Token-level Importance Sampling for Direct Preference Optimization With Estimated Weights
Token级重要性采样，用于使用估计权重做DPO
### 关键字
* LLM
* Importance Sampling
* Preference Learning
### 主要内容
#### DPO问题
DPO 是从Bandit Problem（多臂老虎机）衍生的，其中整个响应被视为单个手臂，忽略了 Token 之间的重要性差异，这可能会影响优化效率，难以实现最优结果。
#### 提出措施

### 文章链接
<a href="./papers/4038_TIS_DPO_Token_level_Impor.pdf">查看PDF</a>
<a href="https://openreview.net/forum?id=oF6e2WwxX0">ICLR链接</a>

### 摘要
Direct Preference Optimization (DPO) has been widely adopted for preference alignment of Large Language Models (LLMs) due to its simplicity and effectiveness. However, DPO is derived as a bandit problem in which the whole response is treated as a single arm, ignoring the importance differences between tokens, which may affect optimization efficiency and make it difficult to achieve optimal results. In this work, we propose that the optimal data for DPO has equal expected rewards for each token in winning and losing responses, as there is no difference in token importance. However, since the optimal dataset is unavailable in practice, we propose using the original dataset for importance sampling to achieve unbiased optimization. Accordingly, we propose a token-level importance sampling DPO objective named TIS-DPO that assigns importance weights to each token based on its reward. Inspired by previous works, we estimate the token importance weights using the difference in prediction probabilities from a pair of contrastive LLMs. We explore three methods to construct these contrastive LLMs: (1) guiding the original LLM with contrastive prompts, (2) training two separate LLMs using winning and losing responses, and (3) performing forward and reverse DPO training with winning and losing responses. Experiments show that TIS-DPO significantly outperforms various baseline methods on harmlessness and helpfulness alignment and summarization tasks. We also visualize the estimated weights, demonstrating their ability to identify key token positions.
直接偏好优化 （DPO） 因其简单性和有效性而被广泛用于大型语言模型 （LLMs。然而，DPO 是作为老虎机问题衍生的，其中整个响应被视为单个手臂，忽略了 Token 之间的重要性差异，这可能会影响优化效率，难以实现最优结果。在这项工作中，我们提出 DPO 的最佳数据在获胜和失败响应中对每个代币的预期奖励相等，因为代币的重要性没有差异。然而，由于实际中没有最优数据集，我们建议使用原始数据集进行重要性采样，以实现无偏优化。因此，我们提出了一个名为 TIS-DPO 的代币级重要性抽样 DPO 目标，该目标根据每个代币的奖励为每个代币分配重要性权重。受以前工作的启发，我们使用一对对比LLMs。我们探索了三种方法来构建这些对比LLMs：（1） 用对比提示引导原始 LLM，（2） 使用获胜和失败响应训练两个单独的 LLMs，以及 （3） 使用获胜和失败响应进行正向和反向 DPO 训练。实验表明，TIS-DPO 在无害性和有用性对齐和总结任务上明显优于各种基线方法。我们还将估计的权重可视化，展示了他们识别关键代币位置的能力。

## 占位
### 关键字
### 主要内容
### 文章链接
<a href="">查看PDF</a>
<a href="">ICLR链接</a>

### 摘要

## 占位
### 关键字
### 主要内容
### 文章链接
<a href="">查看PDF</a>
<a href="">ICLR链接</a>

### 摘要

## 占位
### 关键字
### 主要内容
### 文章链接
<a href="">查看PDF</a>
<a href="">ICLR链接</a>

### 摘要

## 占位
### 关键字
### 主要内容
### 文章链接
<a href="">查看PDF</a>
<a href="">ICLR链接</a>


### 摘要

## 占位
### 关键字
### 主要内容
### 文章链接
<a href="">查看PDF</a>
<a href="">ICLR链接</a>


### 摘要

## 占位
### 关键字
### 主要内容
### 文章链接
<a href="">查看PDF</a>
<a href="">ICLR链接</a>


### 摘要

## 占位
### 关键字
### 主要内容
### 文章链接
<a href="">查看PDF</a>
<a href="">ICLR链接</a>


### 摘要