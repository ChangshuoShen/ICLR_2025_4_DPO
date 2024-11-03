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
Reinforcement learning from human feedback (RLHF) aligns Large Language Models (LLMs) with human preferences. However, these preferences can often change over time due to external factors (e.g. environment change and societal influence). Consequently, what was wrong then might be right now. Current preference optimization algorithms do not account for temporal preference drift in their modeling, which can lead to severe misalignment. To address this limitation, we use a Dynamic Bradley-Terry model that models preferences via time-dependent reward functions, and propose Non-Stationary Direct Preference Optimization (NS-DPO). By introducing a discount parameter in the loss function, NS-DPO applies exponential weighting, which proportionally focuses learning on more time-relevant datapoints. We theoretically analyse the convergence of NS-DPO in the offline setting, providing upper bounds on the estimation error caused by non-stationary preferences. Finally, we demonstrate the effectiveness of NS-DPO1 for fine-tuning LLMs in scenarios with drifting preferences. By simulating preference drift using renowned reward models and modifying popular LLM datasets accordingly, we show that NS-DPO fine-tuned LLMs remain robust under non-stationarity, significantly outperforming baseline algorithms that ignore temporal preference changes, without sacrificing performance in stationary cases.
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
* <a href="./papers/1691_Step_DPO_Step_wise_Prefer.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=H5FUVj0vMd">ICLR链接</a>

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
* 提出建议DPO的最佳数据对于获胜和失败响应中的每个token有相同的预期奖励，因为令牌重要性没有差异
* 但是，因为实际中无法获得最优数据集，所以建议使用原始数据进行重要性采样，以实现无偏优化
#### TIS(Token-level Importance Sampling)-DPO
根据每个令牌的奖励为其分配重要度权重

具体：使用一对"对比性LLM"的预测概率来估计标记重要性权重
1. 使用对比Prompt指导初始LLM
2. 使用获胜和失败的响应训练训练两个独立的LLMs
3. 使用获胜和失败的响应前向和反向的训练DPO
### 文章链接
* <a href="./papers/4038_TIS_DPO_Token_level_Impor.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=oF6e2WwxX0">ICLR链接</a>

### 摘要
Direct Preference Optimization (DPO) has been widely adopted for preference alignment of Large Language Models (LLMs) due to its simplicity and effectiveness. However, DPO is derived as a bandit problem in which the whole response is treated as a single arm, ignoring the importance differences between tokens, which may affect optimization efficiency and make it difficult to achieve optimal results. In this work, we propose that the optimal data for DPO has equal expected rewards for each token in winning and losing responses, as there is no difference in token importance. However, since the optimal dataset is unavailable in practice, we propose using the original dataset for importance sampling to achieve unbiased optimization. Accordingly, we propose a token-level importance sampling DPO objective named TIS-DPO that assigns importance weights to each token based on its reward. Inspired by previous works, we estimate the token importance weights using the difference in prediction probabilities from a pair of contrastive LLMs. We explore three methods to construct these contrastive LLMs: (1) guiding the original LLM with contrastive prompts, (2) training two separate LLMs using winning and losing responses, and (3) performing forward and reverse DPO training with winning and losing responses. Experiments show that TIS-DPO significantly outperforms various baseline methods on harmlessness and helpfulness alignment and summarization tasks. We also visualize the estimated weights, demonstrating their ability to identify key token positions.
直接偏好优化 （DPO） 因其简单性和有效性而被广泛用于大型语言模型 （LLMs。然而，DPO 是作为老虎机问题衍生的，其中整个响应被视为单个手臂，忽略了 Token 之间的重要性差异，这可能会影响优化效率，难以实现最优结果。在这项工作中，我们提出 DPO 的最佳数据在获胜和失败响应中对每个代币的预期奖励相等，因为代币的重要性没有差异。然而，由于实际中没有最优数据集，我们建议使用原始数据集进行重要性采样，以实现无偏优化。因此，我们提出了一个名为 TIS-DPO 的代币级重要性抽样 DPO 目标，该目标根据每个代币的奖励为每个代币分配重要性权重。受以前工作的启发，我们使用一对对比LLMs。我们探索了三种方法来构建这些对比LLMs：（1） 用对比提示引导原始 LLM，（2） 使用获胜和失败响应训练两个单独的 LLMs，以及 （3） 使用获胜和失败响应进行正向和反向 DPO 训练。实验表明，TIS-DPO 在无害性和有用性对齐和总结任务上明显优于各种基线方法。我们还将估计的权重可视化，展示了他们识别关键代币位置的能力。

## 15. Accelerated Preference Optimization for Large Language Model Alignment
Alignment的加速偏好优化
### 关键字
* LLM
* RLHF
* DPO
### 主要内容
#### 提出问题
RLHF能否通过动量技术加速
#### 证明流程
1. 表明迭代优化算法(Iterative Preference Optimization)可以被视为PPO(近端点方法)
2. 提出通用的加速优化(APO)框架，统一现有的许多偏好优化，并采用Nestrov动量技术加速LLM对齐
3. 理论上证明APO可以比标准迭代偏好优化算法(DPO, SPPO等)实现更快的收敛速度

### 文章链接
* <a href="./papers/13320_Accelerated_Preference_O.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=TROUDY6Wg4">ICLR链接</a>

### 摘要
Reinforcement Learning from Human Feedback (RLHF) has emerged as a pivotal tool for aligning large language models (LLMs) with human preferences. Direct Preference Optimization (DPO), one of the most popular approaches, formulates RLHF as a policy optimization problem without explicitly estimating the reward function. It overcomes the stability and efficiency issues of two-step approaches, which typically involve first estimating the reward function and then optimizing the policy via proximal policy optimization (PPO). Since RLHF is essentially an optimization problem, and it is well-known that momentum techniques can accelerate optimization both theoretically and empirically, a natural question arises: Can RLHF be accelerated by momentum? This paper answers this question in the affirmative. In detail, we first show that the iterative preference optimization method can be viewed as a proximal point method. Based on this observation, we propose a general Accelerated Preference Optimization (APO) framework, which unifies many existing preference optimization algorithms and employs Nesterov's momentum technique to speed up the alignment of LLMs. Theoretically, we demonstrate that APO can achieve a faster convergence rate than the standard iterative preference optimization methods, including DPO and SPPO. Empirically, we show the superiority of APO over DPO, iterative DPO, and other strong baselines for RLHF on the AlpacaEval 2.0 benchmark.
人类反馈强化学习 (RLHF) 已成为使大型语言模型 ( LLMs ) 与人类偏好保持一致的关键工具。直接偏好优化 (DPO) 是最流行的方法之一，它将 RLHF 表述为策略优化问题，而无需明确估计奖励函数。它克服了两步方法的稳定性和效率问题，两步方法通常首先估计奖励函数，然后通过近端策略优化（PPO）来优化策略。由于 RLHF 本质上是一个优化问题，而且众所周知，动量技术可以在理论上和经验上加速优化，所以一个自然的问题就出现了：RLHF 可以通过动量加速吗？本文对这个问题给出了肯定的回答。详细地说，我们首先表明迭代偏好优化方法可以被视为近端点方法。基于这一观察，我们提出了一个通用的加速偏好优化（APO）框架，该框架统一了许多现有的偏好优化算法，并采用 Nesterov 的动量技术来加速LLMs的对齐。理论上，我们证明 APO 可以比标准迭代偏好优化方法（包括 DPO 和 SPPO）实现更快的收敛速度。根据经验，我们在 AlpacaEval 2.0 基准上展示了 APO 相对于 DPO、迭代 DPO 和 RLHF 的其他强大基线的优越性。



## 16. Earlier Tokens Contribute More: Learning Direct Preference Optimization From Temporal Decay Perspective
早期token贡献更多：从时间衰减角度学习DPO
### 关键字
* Preference Optimization
* RLHF
* DPO
### 主要内容
#### DPO问题
* DPO存在长度偏差，生成的相应比参考模型的响应更长
* SimPO 和 SamPO 等现有解决方案解决了这个问题，但统一对待跨序列的奖励贡献，忽略了时间动态。
#### 增强的偏好优化方法
* 结合由\参数控制的时间衰减因子
* “动态加权机制”：根据每个奖励在序列中的位置来调整每个奖励的影响，优先考虑对齐更重要的早期标记
#### 效果
通过自适应地关注更相关的反馈，这种方式可以减轻对不太相关的数据的过度拟合，并保持对不断变化的人类偏好的响应
### 文章链接
* <a href="./papers/6333_Earlier_Tokens_Contribute.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=OspqtLVUN5">ICLR链接</a>

### 摘要
Direct Preference Optimization (DPO) has gained attention as an efficient alternative to reinforcement learning from human feedback (RLHF) for aligning large language models (LLMs) with human preferences. Despite its advantages, DPO suffers from a length bias, generating responses longer than those from the reference model. Existing solutions like SimPO and SamPO address this issue but uniformly treat the contribution of rewards across sequences, overlooking temporal dynamics. To this end, we propose an enhanced preference optimization method that incorporates a temporal decay factor controlled by a  parameter. This dynamic weighting mechanism adjusts the influence of each reward based on its position in the sequence, prioritizing earlier tokens that are more critical for alignment. By adaptively focusing on more relevant feedback, our approach mitigates overfitting to less pertinent data and remains responsive to evolving human preferences. Experimental results on several benchmarks show that our approach consistently outperforms vanilla DPO by 5.9-8.8 points on AlpacaEval 2 and 3.3-9.7 points on Arena-Hard across different model architectures and sizes.
直接偏好优化 (DPO) 作为人类反馈强化学习 (RLHF) 的有效替代方案而受到关注，用于使大型语言模型 ( LLMs ) 与人类偏好保持一致。尽管有其优点，DPO 仍存在长度偏差，生成的响应比参考模型的响应更长。 SimPO 和 SamPO 等现有解决方案解决了这个问题，但统一对待跨序列的奖励贡献，忽略了时间动态。为此，我们提出了一种增强的偏好优化方法，该方法结合了由伽玛参数控制的时间衰减因子。这种动态加权机制根据每个奖励在序列中的位置来调整每个奖励的影响，优先考虑对对齐更重要的早期标记。通过自适应地关注更相关的反馈，我们的方法可以减轻对不太相关的数据的过度拟合，并保持对不断变化的人类偏好的响应。多个基准测试的实验结果表明，在不同的模型架构和大小上，我们的方法在 AlpacaEval 2 上始终优于普通 DPO 5.9-8.8 点，在 Arena-Hard 上优于普通 DPO 3.3-9.7 点。



## 17. Step-Controlled DPO: Leveraging Stepwise Errors for Enhancing Mathematical Reasoning of Language Models
利用逐步误差增强语言模型的数学推理(对比13.)
### 关键字
* LLM
* Mathematical Reasoning
* Alignment with relative feedback
### 主要内容
#### SCDPO(步进控制DPO)
通过创建在指定步骤开始出错的数学推理原理的负样本来自动提供逐步错误监督的方法
### 文章链接
* <a href="./papers/1626_Step_Controlled_DPO_Lever.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=ZRDa2IT1sQ">ICLR链接</a>

### 摘要
Direct Preference Optimization (DPO) has proven effective at improving the performance of large language models (LLMs) on downstream tasks such as reasoning and alignment. In this work, we propose Step-Controlled DPO (SCDPO), a method for automatically providing stepwise error supervision by creating negative samples of mathematical reasoning rationales that start making errors at a specified step. By applying these samples in DPO training, SCDPO can better align the model to avoid reasoning errors and output accurate reasoning steps. Qualitative analysis of the credit assignment of SCDPO and DPO demonstrates the effectiveness of SCDPO at identifying errors in mathematical solutions. We then apply SCDPO to an InternLM2-20B model, resulting in a 20B model that achieves competitive scores of 88.5% on GSM8K and 58.1% on MATH, rivaling all other open-source LLMs, showing the great potential of our method. The code, models and data are released to inspire future work.
事实证明，直接偏好优化 (DPO) 可以有效提高大型语言模型 ( LLMs ) 在推理和对齐等下游任务上的性能。在这项工作中，我们提出了步进控制 DPO（SCDPO），这是一种通过创建在指定步骤开始出错的数学推理原理的负样本来自动提供逐步错误监督的方法。通过将这些样本应用于DPO训练，SCDPO可以更好地对齐模型，避免推理错误并输出准确的推理步骤。对 SCDPO 和 DPO 的学分分配的定性分析证明了 SCDPO 在识别数学解决方案中的错误方面的有效性。然后，我们将 SCDPO 应用于 InternLM2-20B 模型，得到的 20B 模型在 GSM8K 上获得了 88.5% 的竞争分数，在 MATH 上获得了 58.1% 的竞争分数，可与所有其他开源LLMs相媲美，显示了我们方法的巨大潜力。代码、模型和数据的发布是为了启发未来的工作。



## 18. MIA-DPO: Multi-Image Augmented Direct Preference Optimization For Large Vision-Language Models
大视觉语言模型的多图像增强DPO
### 关键字
* LVLM(Large Vision Language Models) 
### 主要内容
#### 当前局限性
现有的视觉对齐方法主要针对单图像场景而设计，由于缺乏多样化的训练数据以及注释“接受/拒绝对”成本高，难以有效处理多图像任务的复杂性
#### MIA-DPO
* 通过以`网格拼贴`或`画中画格式排列`的不相关图像来扩展单图像数据，从而缓解了多样化多图像训练数据的稀缺性，从而降低与多图像数据注释相关的成本
* 使用注意力值来识别和过滤掉模型可能错误关注的被拒绝的响应，使用注意力感知选择来构建`选择/拒绝对`而不依赖于人工注释、额外数据、外部模型或API

### 文章链接
* <a href="./papers/1438_MIA_DPO_Multi_Image_Augme.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=f7WBRSuf9l">ICLR链接</a>

### 摘要
Visual preference alignment involves training Large Vision-Language Models (LVLMs) to predict human preferences between visual inputs. This is typically achieved by using labeled datasets of chosen/rejected pairs and employing optimization algorithms like direct preference optimization (DPO). Existing visual alignment methods, primarily designed for single-image scenarios, struggle to effectively handle the complexity of multi-image tasks due to the scarcity of diverse training data and the high cost of annotating chosen/rejected pairs. We present Multi-Image Augmented Direct Preference Optimization (MIA-DPO), a visual preference alignment approach that effectively handles multi-image inputs. MIA-DPO mitigates the scarcity of diverse multi-image training data by extending single-image data with unrelated images arranged in grid collages or pic-in-pic formats, significantly reducing the costs associated with multi-image data annotations. Our observation reveals that attention values of LVLMs vary considerably across different images. We use attention values to identify and filter out rejected responses the model may have mistakenly focused on. Our attention-aware selection for constructing the chosen/rejected pairs without relying on (i) human annotation, (ii) extra data, and (iii) external models or APIs. MIA-DPO is compatible with various architectures and outperforms existing methods on five multi-image benchmarks, achieving an average performance boost of 3.0% on LLaVA-v1.5 and 4.3% on the recent InternLM-XC2.5. Moreover, MIA-DPO has a minimal effect on the model's ability to understand single images.
视觉偏好对齐涉及训练大型视觉语言模型（LVLM）来预测人类在视觉输入之间的偏好。这通常是通过使用所选/拒绝对的标记数据集并采用直接偏好优化 (DPO) 等优化算法来实现的。现有的视觉对齐方法主要针对单图像场景而设计，由于缺乏多样化的训练数据以及注释选择/拒绝对的成本高昂，难以有效处理多图像任务的复杂性。我们提出了多图像增强直接偏好优化（MIA-DPO），这是一种有效处理多图像输入的视觉偏好对齐方法。 MIA-DPO 通过以网格拼贴或画中画格式排列的不相关图像来扩展单图像数据，从而缓解了多样化多图像训练数据的稀缺性，从而显着降低了与多图像数据注释相关的成本。我们的观察表明，不同图像的 LVLM 注意力值差异很大。我们使用注意力值来识别和过滤掉模型可能错误关注的被拒绝的响应。我们的注意力感知选择用于构建所选/拒绝对，而不依赖于 (i) 人工注释、(ii) 额外数据和 (iii) 外部模型或 API。 MIA-DPO 与各种架构兼容，并在五个多图像基准测试中优于现有方法，在 LLaVA-v1.5 上实现了 3.0% 的平均性能提升，在最近的 InternLM-XC2.5 上实现了 4.3% 的平均性能提升。此外，MIA-DPO 对模型理解单个图像的能力影响很小。

## 19. Model Editing as a Robust and Denoised variant of DPO: A Case Study on Toxicity
模型编辑作为DPO的稳健和去噪变体：毒性案例研究
### 关键字
* Model Editing
* Mechanistic Interpretability
* AI Safety
* Alignment
* Toxicity
* LLMs
### 主要内容
#### DPO现有问题
* 计算量大
* 缺乏可控性和透明度
* Tuning-based方法需要大规模的偏好数据进行训练，且容易受到噪声偏好数据的影响
#### ProFS(Projection Filter for Subspaces)
* 免调整对齐替代方案
* 证明其在毒性降低用例下的有效性
* ProFS 基于`因子分析理论`，是一种样本高效的模型编辑方法，可识别模型参数空间中的有毒子空间，并通过投影检测到的子空间来降低模型毒性。
* 通过从语言模型中提取偏好数据嵌入并从这些嵌入中删除无毒信息来识别有毒子空间，证明 ProFS 比 DPO 的样本效率更高，进一步展示了对噪声数据的更强鲁棒性。
* 尝试通过在 ProFS 和 DPO 之间建立理论和经验联系，将基于调整的对齐与编辑联系起来，表明 ProFS 可以解释为`单个 DPO 步骤的去噪版本`!!。

### 文章链接
* <a href="./papers/13296_Model_Editing_as_a_Robus.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=lOi6FtIwR8">ICLR链接</a>

### 摘要
Recent alignment algorithms such as direct preference optimization (DPO) have been developed to improve the safety of large language models (LLMs) by training these models to match human behaviors exemplified by preference data. However, these methods are both computationally intensive and lacking in controllability and transparency, inhibiting their widespread use. Furthermore, these tuning-based methods require large-scale preference data for training and are susceptible to noisy preference data. In this paper, we introduce a tuning-free alignment alternative, ProFS (Projection Filter for Subspaces), and demonstrate its effectiveness under the use case of toxicity reduction. Grounded on theory from factor analysis, ProFS is a sample-efficient model editing approach that identifies a toxic subspace in the model parameter space and reduces model toxicity by projecting away the detected subspace. The toxic subspace is identified by extracting preference data embeddings from the language model, and removing non-toxic information from these embeddings. We show that ProFS is more sample-efficient than DPO, further showcasing greater robustness to noisy data. Finally, we attempt to connect tuning based alignment with editing, by establishing both theoretical and empirical connections between ProFS and DPO, showing that ProFS can be interpreted as a denoised version of a single DPO step.
最近开发的对齐算法，例如直接偏好优化（DPO），通过训练这些模型来匹配偏好数据所例证的人类行为，以提高大型语言模型（ LLMs ）的安全性。然而，这些方法计算量大，缺乏可控性和透明度，限制了它们的广泛使用。此外，这些基于调整的方法需要大规模的偏好数据进行训练，并且容易受到噪声偏好数据的影响。在本文中，我们介绍了一种免调整对齐替代方案 ProFS（子空间投影滤波器），并证明了其在毒性降低用例下的有效性。 ProFS 基于因子分析理论，是一种样本高效的模型编辑方法，可识别模型参数空间中的有毒子空间，并通过投影检测到的子空间来降低模型毒性。通过从语言模型中提取偏好数据嵌入并从这些嵌入中删除无毒信息来识别有毒子空间。我们证明 ProFS 比 DPO 的样本效率更高，进一步展示了对噪声数据的更强鲁棒性。最后，我们尝试通过在 ProFS 和 DPO 之间建立理论和经验联系，将基于调整的对齐与编辑联系起来，表明 ProFS 可以解释为单个 DPO 步骤的去噪版本。


## 20. Learning Dynamics of LLM Finetuning
LLM微调的`学习动态`
### 关键字
* Learning Dynamics(描述特定训练示例的学习如何影响模型对其他示例的预测，为我们提供了理解深度学习系统行为的工具)
* LLM
* Fine Tuning 
* DPO
### 主要内容
#### 学习动态
学习动态描述了特定训咯实例的学习如何影响模型对其他示例的预测，帮助理解深度学习系统行为
#### 做了什么
1. 通过分析不同潜在反应之间影响力如何累积的逐步分解，研究大型语言模型在不同类型的微调过程中的学习动态。
2. 提出的框架允许对有关指令调整和偏好调整的流行算法训练的许多有趣的观察结果进行统一解释。
3. 提出了为什么特定类型的幻觉在微调后得到加强的假设解释，例如，模型可能会在问题 B 的回答中使用短语或事实来回答问题 A，或者模型在生成问题时可能会不断重复类似的简单短语回应。
4. 扩展上述框架，并强调了一种独特的“挤压效应”，以解释之前在off-policy DPO中观察到的现象，即运行 DPO 时间过长甚至会导致所需输出的可能性降低。
5. 该框架还帮助理解on-policy DPO和其他变体的优势
### 文章链接
* <a href="./papers/4818_Learning_Dynamics_of_LLM_.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=tPNHOoZFl9">ICLR链接</a>

### 摘要
Learning dynamics, which describes how the learning of specific training examples influences the model's predictions on other examples, gives us a powerful tool for understanding the behavior of deep learning systems. We study the learning dynamics of large language models during different types of finetuning, by analyzing the step-wise decomposition of how influence accumulates among different potential responses. Our framework allows a uniform interpretation of many interesting observations about the training of popular algorithms for both instruction tuning and preference tuning. In particular, we propose a hypothetical explanation of why specific types of hallucination are strengthened after finetuning, e.g., the model might use phrases or facts in the response for question B to answer question A, or the model might keep repeating similar simple phrases when generating responses. We also extend our framework and highlight a unique ``squeezing effect'' to explain a previously observed phenomenon in off-policy direct preference optimization (DPO), where running DPO for too long makes even the desired outputs less likely. This framework also provides insights into where the benefits of on-policy DPO and other variants come from. The analysis not only provides a novel perspective of understanding LLM's finetuning but also inspires a simple, effective method to improve alignment performance.
学习动态描述了特定训练示例的学习如何影响模型对其他示例的预测，为我们提供了理解深度学习系统行为的强大工具。我们通过分析不同潜在反应之间影响力如何累积的逐步分解，研究大型语言模型在不同类型的微调过程中的学习动态。我们的框架允许对有关指令调整和偏好调整的流行算法训练的许多有趣的观察结果进行统一解释。特别是，我们提出了为什么特定类型的幻觉在微调后得到加强的假设解释，例如，模型可能会在问题 B 的回答中使用短语或事实来回答问题 A，或者模型在生成问题时可能会不断重复类似的简单短语回应。我们还扩展了我们的框架，并强调了一种独特的“挤压效应”，以解释之前在离策略直接偏好优化 (DPO) 中观察到的现象，即运行 DPO 时间过长甚至会导致所需输出的可能性降低。该框架还提供了有关同保单 DPO 和其他变体的优势从何而来的见解。该分析不仅提供了理解LLM微调的新颖视角，而且启发了一种简单、有效的方法来提高对齐性能。


## 21. On the Generalization of Preference Learning with DPO
关于 DPO 偏好学习的泛化
### 关键字
* Preference Learning
* Generalization Bound(泛化界限)
### 主要内容
#### 面对的问题
人们对这些模型的泛化保证仍然缺乏透彻的理论认识
#### 这篇工作做了什么
* 提供一个新的理论框架来分析DPO训练的泛化保证，从而弥补上述差距
* 现有的泛化理论通常侧重于实现接近最优损失的过参数化模型或独立于训练过程的模型
* 这篇工作的框架则严格评估`模型在经过有限数量的梯度步骤后的泛化效果`，这反映了现实世界中的 LLM 训练实践。
* 通过分析与整个样本及其整个训练轨迹相关的奖励边际，可以有效限制泛化误差
* 其推导的学习保证表明，在特定条件下，使用DPO的模型能以极高的概率在未见数据上分辨出prefered响应

### 文章链接
* <a href="./papers/10640_On_the_Generalization_of.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=bGkPZtisSm">ICLR链接</a>

### 摘要
Large language models (LLMs) have demonstrated remarkable capabilities but often struggle to align with human preferences, leading to harmful or undesirable outputs. Preference learning, which trains models to distinguish between preferred and non-preferred responses based on human feedback, has become a crucial component for ensuring that LLMs align with human values. Despite the widespread adoption in real-world systems, a thorough theoretical understanding of the generalization guarantees for these models remains lacking. This paper bridges that gap by introducing a new theoretical framework to analyze the generalization guarantees of models trained with direct preference optimization. While existing generalization theory often focuses on overparameterized models achieving near-optimal loss or models independent of the training process, our framework rigorously assesses how well models generalize after a finite number of gradient steps, reflecting real-world LLM training practices. By analyzing the reward margin associated with each sample and its trajectory throughout training, we can effectively bound the generalization error. We derive learning guarantees showing that, under specific conditions, models trained with DPO can correctly discern preferred responses on unseen data with high probability. These insights are empirically validated on contemporary LLMs, underscoring the practical relevance of our theory.
大型语言模型 ( LLMs ) 已展现出卓越的功能，但往往难以与人类偏好保持一致，从而导致有害或不良的输出。偏好学习训练模型根据人类反馈区分偏好和非偏好反应，已成为确保LLMs符合人类价值观的关键组成部分。尽管在现实系统中得到广泛采用，但仍然缺乏对这些模型的泛化保证的全面理论理解。本文通过引入新的理论框架来分析直接偏好优化训练的模型的泛化保证，从而弥补了这一差距。虽然现有的泛化理论通常侧重于实现接近最优损失的过度参数化模型或独立于训练过程的模型，但我们的框架严格评估模型在有限数量的梯度步骤后的泛化程度，反映了现实世界的LLM培训实践。通过分析与每个样本相关的奖励裕度及其在整个训练过程中的轨迹，我们可以有效地限制泛化误差。我们得出的学习保证表明，在特定条件下，使用 DPO 训练的模型能够以高概率正确识别对未见数据的首选响应。这些见解在当代LLMs中得到了实证验证，强调了我们理论的实际相关性。

## 22. Anchored Alignment for Self-Explanations Enhancement
增强自我解释的锚定对齐
### 关键字
* LLM
* Self-Explaination(The model’s ability to generate explanations for its outputs, enhancing transparency and trust.)
* Alignment
* Preference Pairs
* DPO
* SFT(Supervised Fine-Tuning)
* RLAIF(Reinforcement Learning from AI Feedback)
* Self-Alignment(When an LLM is fine-tuned to follow guidelines or instructions it helped create, ensuring internal consistency.)
* Self-Instruction(Teaching the LLM to generate instructions or examples to improve its own learning and accuracy in responses.)
### 主要内容
#### 新对其方法
旨在增强LLM的自我解释能力——即使在没有带注释的基本原理解释的情况下也是如此
#### 细节——三个关键组成部分
* 解释质量评估
* 自引导数据集生成
* 模型对齐
#### 新技术：Alignment with Anchor Preference Pairs
通过将模型分为三组来改进偏号集的选择：
* 一致正确
* 一致错误
* 可变

通过对三种类别分别应用量身定制的策略来提高DPO的有效性
### 相关链接
* <a href="./papers/12006_Anchored_Alignment_for_S.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=mkE9Yx4wHY">ICLR链接</a>

### 摘要
In this work, we introduce a methodology for alignment designed to enhance the ability of large language models (LLMs) to articulate their reasoning—\textit{self-explanation}—even in the absence of annotated rationale explanations. Our alignment methodology comprises three key components: explanation quality assessment, self-instruction dataset generation, and model alignment. Additionally, we present a novel technique called \textit{Alignment with Anchor Preference Pairs}, which improves the selection of preference pairs by categorizing model outputs into three groups: consistently correct, consistently incorrect, and variable. By applying tailored strategies to each category, we enhance the effectiveness of Direct Preference Optimization (DPO). Our experimental results demonstrate that this approach significantly improves explanation quality while maintaining accuracy compared to other fine-tuning strategies.
在这项工作中，我们引入了一种对齐方法，旨在增强大型语言模型（ LLMs ）阐明其推理的能力——\textit{自我解释}——即使在没有带注释的基本原理解释的情况下也是如此。我们的对齐方法包括三个关键组成部分：解释质量评估、自指导数据集生成和模型对齐。此外，我们提出了一种名为 \textit{Alignment with Anchor Preference Pairs} 的新技术，它通过将模型输出分为三组来改进偏好对的选择：一致正确、一致错误和可变。通过对每个类别应用量身定制的策略，我们提高了直接偏好优化 (DPO) 的有效性。我们的实验结果表明，与其他微调策略相比，这种方法显着提高了解释质量，同时保持了准确性。

## 23. Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization
无意的不一致：DPO中的可能性位移
### 关键字
* DPO
* Likelihood Displacement
* Unalignment & Alignment
* Language Models 

### 主要内容
#### DPO及其变体的问题
先前的工作观察到训练期间的Prefered Response的可能性总会降低
#### 这篇工作的`结论`
1. 将这种现象称为`Likelihood Displacement`
2. 这种现象可能是灾难性的，会将概率质量从Prefered Response转到语义相反的Response上
3. 在调整模型拒绝一些不安全的prompt的时候，发现这种Displacement会将概率质量从首选的拒绝反应转向有害反应，无意中导致Unalignment
4. 从理论上描述Likelihood Displacement是由诱导相似嵌入的偏好驱动的，这是由居中隐藏嵌入相似性（CHES， centered hidden embedding similarity）得分来衡量的。
5. 强调以足够独特的偏好来整理数据的重要性
### 相关链接
* <a href="./papers/7503_Unintentional_Unalignment.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=uaMSBJDnRv">ICLR链接</a>

### 摘要
Direct Preference Optimization (DPO), and its numerous variants, are increasingly used for aligning language models. Although they are designed to teach a model to generate preferred responses more frequently relative to dispreferred responses, prior work has observed that the likelihood of preferred responses often decreases during training. The current work sheds light on the causes and implications of this counter-intuitive phenomenon, which we term likelihood displacement. We demonstrate that likelihood displacement can be catastrophic, shifting probability mass from preferred responses to semantically opposite ones. As a simple example, training a model to prefer over can sharply increase the probability of . Moreover, when aligning the model to refuse unsafe prompts, we show that such displacement can unintentionally lead to unalignment, by shifting probability mass from preferred refusal responses to harmful responses (e.g., reducing the refusal rate of Llama-3-8B-Instruct from 74.4% to 33.4%). We theoretically characterize that likelihood displacement is driven by preferences that induce similar embeddings, as measured by a centered hidden embedding similarity (CHES) score. Empirically, the CHES score enables identifying which training samples contribute most to likelihood displacement in a given dataset. Filtering out these samples effectively mitigated unintentional unalignment in our experiments. More broadly, our results highlight the importance of curating data with sufficiently distinct preferences, for which we believe the CHES score may prove valuable.
直接偏好优化 (DPO) 及其众多变体越来越多地用于对齐语言模型。尽管它们的目的是教导模型相对于不良反应更频繁地生成首选反应，但先前的工作已经观察到，在训练期间，首选反应的可能性通常会降低。目前的工作揭示了这种反直觉现象的原因和影响，我们将其称为似然位移。我们证明，似然位移可能是灾难性的，将概率质量从首选响应转移到语义相反的响应。举一个简单的例子，训练一个模型以使其更喜欢 超过 可以急剧增加概率 。此外，当调整模型以拒绝不安全提示时，我们表明，通过将概率质量从首选拒绝响应转移到有害响应（例如，将 Llama-3-8B-Instruct 的拒绝率从 74.4 降低），这种位移可能会无意中导致不对齐 。 % 至 33.4%）。我们从理论上描述了似然位移是由引起相似嵌入的偏好驱动的，通过中心隐藏嵌入相似性（CHES）得分来衡量。根据经验，CHES 分数能够识别哪些训练样本对给定数据集中的似然位移贡献最大。过滤掉这些样本有效地减轻了我们实验中无意的未对齐情况。更广泛地说，我们的结果强调了以足够独特的偏好来整理数据的重要性，我们相信 CHES 分数可能会证明这一点很有价值。


## 24. Combating inherent noise for direct preference optimization
对抗固有噪音以实现DPO
### 关键字
* DPO
### 主要内容
#### DPO训练的问题
* DPO训练中使用的偏好数据的质量很大程度上被忽视了
* 当前的数据集都包含噪音标签
#### 技术细节
* 将噪音感知指标加入DPO目标中
    * 注释者内部置信度
    * 注视者间稳定性
有助于识别和减轻噪音数据的影响
* 引入Adaptive-DPO损失函数，通过两种方式改善DPO损失
    * 减少噪声样本的影响
    * 放大干净样本的影响

### 相关链接
* <a href="./papers/2094_Combating_inherent_noise_.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=MlxeUVCQgD">ICLR链接</a>

### 摘要
Direct Preference Optimization (DPO) has recently gained traction as a promising approach to align large models with human feedback. It is notable for its effectiveness and ease of application across various models, including Large Language Models (LLMs) and Diffusion Models (DMs). However, the quality of preference data used in DPO training has been largely overlooked. Current datasets, whether annotated by deep learning metrics or crowd-sourced human judgments, often contain noisy labels. This noise can adversely affect the performance of DPO. To address this issue, we propose a novel approach that incorporates a noise-aware metric into the DPO objective. This metric, which includes intra-annotator confidence and inter-annotator stability, helps identify and mitigate the impact of noisy data. We introduce an Adaptive-DPO loss function which improves the DPO loss in two ways: one aims to reduce the influence of noisy samples, while the other is to amplify the impact of clean samples. Our experiments demonstrate that this method effectively handles both synthetic and natural noisy data, leading to improved performance in visual and textual generation tasks. This underscores the practical value of our approach in enhancing model robustness amidst noisy preference data.
直接偏好优化（DPO）最近作为一种将大型模型与人类反馈结合起来的有前途的方法而受到关注。它以其在各种模型中的有效性和易用性而闻名，包括大型语言模型 ( LLMs ) 和扩散模型 (DM)。然而，DPO 培训中使用的偏好数据的质量在很大程度上被忽视了。当前的数据集，无论是通过深度学习指标还是众包的人类判断来注释，通常都包含嘈杂的标签。这种噪音会对 DPO 的性能产生不利影响。为了解决这个问题，我们提出了一种新颖的方法，将噪声感知指标纳入 DPO 目标中。该指标包括注释者内部置信度和注释者间稳定性，有助于识别和减轻噪声数据的影响。我们引入了 Adaptive-DPO 损失函数，它通过两种方式改善 DPO 损失：一是减少噪声样本的影响，二是放大干净样本的影响。我们的实验表明，该方法可以有效处理合成和自然噪声数据，从而提高视觉和文本生成任务的性能。这强调了我们的方法在嘈杂的偏好数据中增强模型鲁棒性的实用价值。


## 25. On Extending Direct Preference Optimization to Accommodate Ties
关于扩展DPO以适应平局
### 关键字
* Preference Optimization
* Ties(此处应该是指平局)
* DPO
* Language Model
* Machine Translation
* Summarization

### 主要内容
#### DPO变体
推导并研究两个DPO变体，明确模拟了在成对比较中声明平局的可能性
* 将DPO中Bradley-Terry模型换成Rao-Kupper模型以及Davison中的两个著名的扩展建模，将平局概率作为明确偏好的替代方案
    * Bradley-Terry模型：概率模型，用于评估成对的胜出模型，不允许平局，只有一方胜出$P(i > j) = \frac{p_i}{p_i + p_j}$
    * Rao-Kupper模型：上面模型的扩展，考虑平局的情况$p(i > j) = \frac{p_i}{p_i + p_j + \delta}; p(i = j) = \frac{\delta}{p_i + p_j + \delta}$
    * Davidson模型：进一步扩展，适用于有高频率平局的情况$p(i > j) = \frac{p_i}{p_i + p_j}(1 - \delta); p(i = i) = \delta $
* 在翻译和摘要的任务上验证了可以将显示标记关系添加到这些DPO变体的数据集中，而不会出现将相同的关系呈现给DPO时观察到的任务性能下降
* `凭经验发现`，包含关系会导致相对于通过KL散度衡量的参考策略具有更强的正则性

### 相关链接
* <a href="./papers/3051_On_Extending_Direct_Prefe.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=h71cSd2loX">ICLR链接</a>

### 摘要
We derive and investigate two DPO variants that explicitly model the possibility of declaring a tie in pair-wise comparisons. We replace the Bradley-Terry model in DPO with two well-known modeling extensions, by Rao and Kupper and by Davidson, that assign probability to ties as alternatives to clear preferences. Our experiments in neural machine translation and summarization show that explicitly labeled ties can be added to the datasets for these DPO variants without the degradation in task performance that is observed when the same tied pairs are presented to DPO. We find empirically that the inclusion of ties leads to stronger regularization with respect to the reference policy as measured by KL divergence, and we see this even for DPO in its original form. These findings motivate and enable the inclusion of tied pairs in preference optimization as opposed to simply discarding them.
我们推导并研究了两个 DPO 变体，它们明确模拟了在成对比较中声明平局的可能性。我们将 DPO 中的 Bradley-Terry 模型替换为 Rao 和 Kupper 以及 Davidson 的两个著名的建模扩展，它们将概率分配给关系作为明确偏好的替代方案。我们在神经机器翻译和摘要方面的实验表明，可以将显式标记的关系添加到这些 DPO 变体的数据集中，而不会出现将相同的关系对呈现给 DPO 时观察到的任务性能下降。我们凭经验发现，包含关系会导致相对于通过 KL 散度衡量的参考策略更强的正则化，即使对于原始形式的 DPO，我们也看到了这一点。这些发现激励并使得将绑定对纳入偏好优化中而不是简单地丢弃它们。


## 26. Hybrid Preference Optimization: Augmenting Direct Preference Optimization with Auxiliary Objectives
混合偏好优化：通过辅助目标增强DPO
### 关键字
* LLM
* Alignment
* RL
* DPO

### 主要内容
#### DPO问题
* 虽然 DPO 提供了基于最大似然估计(MLE)的更简单的框架，但它损害了调整语言模型的能力，以便根据LLM设计者的偏好轻松最大化不可微目标
* 这些可能既不符合用户偏好，甚至也无法通过二进制偏好数据轻松捕获。
#### HPO：结合DPO和RL
为了利用DPO的简单性和性能以及RL的通用型，提出了一种 DPO 和 RLHF 之间的混合方法。通过`对 DPO 的隐式奖励分解`进行简单的增强，允许调整LLMs ，以使用离线 RL 最大化一组任意辅助奖励。

### 相关链接
* <a href="./papers/4013_Hybrid_Preference_Optimiz.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=F5nWSf9etp">ICLR链接</a>

### 摘要
For aligning large language models (LLMs), prior work has leveraged reinforcement learning via human feedback (RLHF) or variations of direct preference optimization (DPO). While DPO offers a simpler framework based on maximum likelihood estimation, it compromises on the ability to tune language models to easily maximize non-differentiable objectives according to the LLM designer's preferences (e.g., using simpler language or minimizing specific kinds of harmful content). These may neither align with user preferences nor even be able to be captured tractably by binary preference data. To leverage the simplicity and performance of DPO with the generalizability of RL, we propose a hybrid approach between DPO and RLHF. With a simple augmentation to the implicit reward decomposition of DPO, we allow for tuning LLMs to maximize a set of arbitrary auxiliary rewards using offline RL. The proposed method, Hybrid Preference Optimization (HPO), shows the ability to effectively generalize to both user preferences and auxiliary designer objectives, while preserving alignment performance across a range of challenging benchmarks and model sizes.
为了调整大型语言模型（ LLMs ），先前的工作通过人类反馈（RLHF）或直接偏好优化（DPO）的变体利用强化学习。虽然 DPO 提供了基于最大似然估计的更简单的框架，但它损害了调整语言模型的能力，以便根据LLM设计者的偏好轻松最大化不可微目标（例如，使用更简单的语言或最小化特定类型的有害内容）。这些可能既不符合用户偏好，甚至也无法通过二进制偏好数据轻松捕获。为了利用 DPO 的简单性和性能以及 RL 的通用性，我们提出了一种 DPO 和 RLHF 之间的混合方法。通过对 DPO 的隐式奖励分解进行简单的增强，我们允许调整LLMs ，以使用离线 RL 最大化一组任意辅助奖励。所提出的方法，混合偏好优化（HPO），显示了有效概括用户偏好和辅助设计者目标的能力，同时在一系列具有挑战性的基准和模型大小中保持对齐性能。


## 27. Bridging and Modeling Correlations in Pairwise Data for Direct Preference Optimization
用于DPO的成对数据中的桥接和建模相关性
### 关键字
* LLM
* Alignment
* Preference optimization

### 主要内容
#### DPO数据集的问题
成对数据中的winning response和losing response是单独生成的，导致它们之间的相关性较弱以及对齐性能不佳
#### 此工作针对数据生成
提出一种有效的成对数据桥接和建模相关性框架，BMC(Bridging & Modeling Correlations)

具体细节：
1. 通过有针对性的修改来提高成对偏好信号的一致性和信息量，以获胜响应为参考，通过改进失败响应来合成伪获胜响应。
2. 发现仅 DPO 不足以对这些相关性进行建模并捕获细微的变化。因此，建议通过在训练期间动态利用策略模型的置信度来学习token级相关性。
### 相关链接
* <a href="./papers/6425_Bridging_and_Modeling_Cor.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=hRwxZmcvW9">ICLR链接</a>

### 摘要
Direct preference optimization (DPO), a widely adopted offline preference optimization algorithm, aims to align large language models (LLMs) with human-desired behaviors using pairwise preference data. However, the winning response and the losing response within pairwise data are generated isolatedly, leading to weak correlations between them as well as suboptimal alignment performance. To address this issue, we propose an effective framework for Bridging and Modeling Correlations in pairwise data, named BMC. Firstly, we increase the consistency and informativeness of the pairwise preference signals through targeted modifications, synthesizing a pseudo-winning response by improving the losing response with the winning response as a reference. Secondly, we identify that DPO alone is insufficient to model these correlations and capture nuanced variations. Therefore, we propose learning token-level correlations by dynamically leveraging the policy model's confidence during training. Comprehensive experiments on QA, math, and instruction-following tasks demonstrate the effectiveness of our approach, significantly surpassing competitive baselines, including DPO. Additionally, our in-depth quantitative analysis reveals the reasons behind our method's superior performance over DPO and showcases its versatility to other DPO variants.
直接偏好优化（DPO）是一种广泛采用的离线偏好优化算法，旨在使用成对偏好数据将大型语言模型（ LLMs ）与人类期望的行为结合起来。然而，成对数据中的获胜响应和失败响应是单独生成的，导致它们之间的相关性较弱以及对齐性能不佳。为了解决这个问题，我们提出了一种有效的成对数据桥接和建模相关性框架，名为 BMC。首先，我们通过有针对性的修改来提高成对偏好信号的一致性和信息量，以获胜响应为参考，通过改进失败响应来合成伪获胜响应。其次，我们发现仅 DPO 不足以对这些相关性进行建模并捕获细微的变化。因此，我们建议通过在训练期间动态利用策略模型的置信度来学习令牌级相关性。关于 QA、数学和指令遵循任务的综合实验证明了我们方法的有效性，显着超越了包括 DPO 在内的竞争基准。此外，我们深入的定量分析揭示了我们的方法优于 DPO 的原因，并展示了其与其他 DPO 变体的多功能性。

## 28. *RainbowPO: A Unified Framework for Combining Improvements in Preference Optimization
RainbowPO：结合偏好优化改进的统一框架
### 关键字
* Alignment
* Preference Optimization
* RLHF
### 主要内容
#### DPO类算法的问题
* 虽然这些方法成功地将模型与人类偏好结合起来，但对其附加组件的贡献缺乏了解。
* 公平和一致的比较很少，因此很难辨别哪些组件真正提高了下游性能。
#### RainbowPO: Unified Framework
* 这是一个统一的框架，通过将现有 DPO 方法的关键组件分类为七个主要方向，揭开了现有 DPO 方法的有效性
* 将这些组件集成到一个单一的有凝聚力的目标中，从而提高每个单独元素的性能。

### 相关链接
* <a href="./papers/5650_RainbowPO_A_Unified_Frame.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=trKee5pIFv">ICLR链接</a>

### 摘要
Recently, numerous preference optimization algorithms have been introduced as extensions to the Direct Preference Optimization (DPO) family. While these methods have successfully aligned models with human preferences, there is a lack of understanding regarding the contributions of their additional components. Moreover, fair and consistent comparisons are scarce, making it difficult to discern which components genuinely enhance downstream performance. In this work, we propose RainbowPO, a unified framework that demystifies the effectiveness of existing DPO methods by categorizing their key components into seven broad directions. We integrate these components into a single cohesive objective, enhancing the performance of each individual element. Through extensive experiments, we demonstrate that RainbowPO outperforms existing DPO variants. Additionally, we provide insights to guide researchers in developing new DPO methods and assist practitioners in their implementations.
最近，许多偏好优化算法被引入作为直接偏好优化 (DPO) 系列的扩展。虽然这些方法成功地将模型与人类偏好结合起来，但对其附加组件的贡献缺乏了解。此外，公平和一致的比较很少，因此很难辨别哪些组件真正提高了下游性能。在这项工作中，我们提出了 RainbowPO，这是一个统一的框架，通过将现有 DPO 方法的关键组件分类为七个主要方向，揭开了现有 DPO 方法有效性的神秘面纱。我们将这些组件集成到一个单一的有凝聚力的目标中，从而提高每个单独元素的性能。通过大量的实验，我们证明 RainbowPO 的性能优于现有的 DPO 变体。此外，我们还提供见解来指导研究人员开发新的 DPO 方法并协助从业者实施。



## 29. Direct Preference Optimization With Unobserved Preference Heterogeneity
具有未观察到的偏好异质性的DPO
### 关键字
* RLHF
* LLM Alignment
* Preference Aggregation(偏好聚合)

### 主要内容
#### 问题
RLHF和DPO都假设统一的偏好，忽视了不同人类的注释者的现实
#### 提出措施 
1. 将生成模型和不同的人类偏好相结合：对DPO的期望最大化适应，根据注释者的潜在偏好类型生成模型的混合
2. 然后引入最小-最大遗憾集成学习模型来产生单一生成方法，以最大限度地减少具有相似潜在因素的注释者子组中最坏情况的遗憾
### 相关链接
* <a href="./papers/12279_Direct_Preference_Optimi.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=NQZNNUsutn">ICLR链接</a>

### 摘要
RLHF has emerged as a pivotal step in aligning language models with human objectives and values. It typically involves learning a reward model from human preference data and then using reinforcement learning to update the generative model accordingly. Conversely, Direct Preference Optimization (DPO) directly optimizes the generative model with preference data, skipping reinforcement learning. However, both RLHF and DPO assume uniform preferences, overlooking the reality of diverse human annotators. This paper presents a new method to align generative models with varied human preferences. We propose an Expectation-Maximization adaptation to DPO, generating a mixture of models based on latent preference types of the annotators. We then introduce a min-max regret ensemble learning model to produce a single generative method to minimize worst-case regret among annotator subgroups with similar latent factors. Our algorithms leverage the simplicity of DPO while accommodating diverse preferences. Experimental results validate the effectiveness of our approach in producing equitable generative policies.
RLHF 的出现是使语言模型与人类目标和价值观保持一致的关键一步。它通常涉及从人类偏好数据中学习奖励模型，然后使用强化学习来相应地更新生成模型。相反，直接偏好优化（DPO）直接使用偏好数据优化生成模型，跳过强化学习。然而，RLHF 和 DPO 都假设统一的偏好，忽视了不同人类注释者的现实。本文提出了一种将生成模型与不同的人类偏好相结合的新方法。我们提出了对 DPO 的期望最大化适应，根据注释者的潜在偏好类型生成模型的混合。然后，我们引入最小-最大遗憾集成学习模型来产生单一生成方法，以最大限度地减少具有相似潜在因素的注释者子组中最坏情况的遗憾。我们的算法利用 DPO 的简单性，同时适应不同的偏好。实验结果验证了我们的方法在制定公平的生成政策方面的有效性。




## 30. Improving Reasoning Ability of Large Language Models via Iterative Uncertainty-based Preference Optimization
通过基于迭代不确定性的偏好优化提高大型语言模型的推理能力
### 关键字
* Preference Optimization
* LLM
* Iterative Optimization 
* Uncertainty 

### 主要内容
#### DPO训练的问题
* 缺少构造高质量的偏好数据集，需要昂贵的人工注释或强大的LM注释
* DPO在复杂推理任务中表现出次优性能
#### 此工作的解决方式
1. 引入一种通过迭代采样和执行反馈来收集偏好对的方法，该方法适用于策略模型当前的学习状态(学习良好、错误学习和未学习)
2. 提出IUPO(Interative Uncertainty-based Preference Optimization)
    * 一种基于迭代不确定性的偏好优化方法，通过评估模型置信度来实现细粒度的偏好控制
    * 为减轻DPO的失败并提高其在推理任务中的适用性
### 相关链接
* <a href="./papers/9446_Improving_Reasoning_Abili.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=bGGMLWAGMc">ICLR链接</a>

### 摘要
Direct Preference Optimization (DPO) has recently emerged as an efficient and effective method for aligning large language models with human preferences. However, constructing high-quality preference datasets remains challenging, often necessitating expensive manual or powerful LM annotations. Additionally, standard DPO exhibits suboptimal performance in complex reasoning tasks, such as mathematical and code reasoning. In this paper, we introduce an approach to collect preference pairs through iterative sampling and execution feedback, tailored to the current learning state (e.g. well-learned, mis-learned, and unlearned) of the policy model. To alleviate the failures of DPO and improve its applicability in reasoning tasks, we propose IUPO, an iterative uncertainty-based preference optimization method that achieves fine-grained preference control by assessing model confidence. We validate our approach across three reasoning tasks, incorporating five established reasoning datasets and one self-curated dataset. Our experimental results demonstrate an overall improvement of 3.6% over the standard DPO method. Furthermore, our approach exhibits promising generalizability involving weak-to-strong (8B to 70B) and cross-model (Llama to Mistral) generalizations.
直接偏好优化（DPO）最近已成为一种使大型语言模型与人类偏好保持一致的高效方法。然而，构建高质量的偏好数据集仍然具有挑战性，通常需要昂贵的手动或强大的 LM 注释。此外，标准 DPO 在复杂推理任务（例如数学和代码推理）中表现出次优性能。在本文中，我们介绍了一种通过迭代采样和执行反馈来收集偏好对的方法，该方法适合策略模型当前的学习状态（例如，学习良好、错误学习和未学习）。为了减轻 DPO 的失败并提高其在推理任务中的适用性，我们提出了 IUPO，一种基于迭代不确定性的偏好优化方法，通过评估模型置信度来实现细粒度的偏好控制。我们在三个推理任务中验证了我们的方法，其中包含五个已建立的推理数据集和一个自行管理的数据集。我们的实验结果表明，与标准 DPO 方法相比，整体性能提高了 3.6%。此外，我们的方法表现出有希望的泛化性，涉及弱到强（8B 到 70B）和跨模型（Llama 到 Mistral）泛化。


## 31. MallowsPO: Fine-Tune Your LLM with Preference Dispersions
通过偏好分散微调LLM
### 关键字
* LLM Fine-Tuning
* Learning from Human Feedback
* Human Preference Dispersions(人类偏好离散度)
    * 指的是在人类偏好评估中观察到的分歧或差异，尤其是在模型训练或偏好优化时。
    * 这种偏好分散性在大模型微调中尤其关键，因为它反映了人类偏好的多样性。理解和建模这种差异可以帮助优化模型，使其输出更好地适应多样化的人类需求

### 主要内容
#### DPO的一个缺点
缺乏描述人类偏好多样性的能力
#### Mallows的由来：Mallows偏好排序理论

### 相关链接
* <a href="./papers/5314_MallowsPO_Fine_Tune_Your_.pdf">查看PDF</a>
* <a href="https:分散性在大模型微调中尤其关键，因为它反映了人类偏好的多样性。理解和建模这种差异可以帮助优化模型，使其输出更好地适应多样化的人类需求

### 主要内容
#### DPO的一个缺点
缺乏描述人类偏好多样性的能力
#### Mallows的由来：Mallows偏好排序理论
一种统计模型，用于描述和分析具有一定偏好顺序的群体排序数据。该理论通过马洛斯模型（Mallows Model）来解释人们对一组对象的排序偏好，并用来计算个体排序与“中心”或“理想”排序的接近程度。其核心参数控制偏好的一致性程度，值越小则群体内偏好越趋于一致。
#### MallowsPO细节
* 使用分散指数(dispersion index)反映人类对提示的偏好的分散性
* 证明当前的DPO模型可以表示为该离散指数的特殊情况，从而与MallowsPO统一
* 凭经验证明了如何使用这种分散指数来提高 DPO 在各种基准任务中的性能，从合成老虎机选择到可控生成和对话，同时保持强大的泛化能力。

### 相关链接
* <a href="./papers/5314_MallowsPO_Fine_Tune_Your_.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=d8cnezVcaW">ICLR链接</a>

### 摘要
Direct Preference Optimization (DPO) has recently emerged as a popular approach to improve reinforcement learning with human feedback (RLHF), leading to better techniques to fine-tune large language models (LLM). A weakness of DPO, however, lies in its lack of capability to characterize the diversity of human preferences. Inspired by Mallows' theory of preference ranking, we develop in this paper a new approach, the MallowsPO. A distinct feature of this approach is a dispersion index, which reflects the dispersion of human preference to prompts. We show that existing DPO models can be reduced to special cases of this dispersion index, thus unified with MallowsPO. More importantly, we demonstrate empirically how to use this dispersion index to enhance the performance of DPO in a broad array of benchmark tasks, from synthetic bandit selection to controllable generation and dialogues, while maintaining great generalization capabilities. MallowsPO is also compatible with other SOTA offline preference optimization methods, boosting nearly 2% extra LC win rate when used as a plugin for fine-tuning Llama3-Instruct.
直接偏好优化（DPO）最近成为一种通过人类反馈（RLHF）改进强化学习的流行方法，从而产生更好的技术来微调大型语言模型（ LLM ）。然而，DPO 的弱点在于它缺乏描述人类偏好多样性的能力。受 Mallows 偏好排序理论的启发，我们在本文中开发了一种新方法，即 MallowsPO 。这种方法的一个显着特征是分散指数，它反映了人类对提示的偏好的分散性。我们证明现有的 DPO 模型可以简化为该离散指数的特殊情况，从而与 MallowsPO 统一。更重要的是，我们凭经验证明了如何使用这种分散指数来提高 DPO 在各种基准任务中的性能，从合成老虎机选择到可控生成和对话，同时保持强大的泛化能力。 MallowsPO 还兼容其他 SOTA 离线偏好优化方法，当用作微调 Llama3-Instruct 的插件时，可额外提高近 2% 的 LC 获胜率。


## 32. Mitigating Forgetting in LLM Supervised Fine-Tuning and Preference Learning
减少LLM监督微调和偏好学习中的遗忘
### 关键字
* Optimization Trade-off
* LLMs
* Supervised Fine-tuning (SFT)
* RLHF

### 主要内容
预训练LLMs的后续训练由`SFT`和`Preference Learning`组成
#### SFT-PO Trade-off
* 就 SFT 和 RLHF/DPO 权衡而言，顺序训练并不是最优的： LLM在进行第二阶段的训练时逐渐忘记了第一阶段的训练。
* 理论上证明了顺序后训练的`次优性`
* 提出一种实用的联合后训练框架，该框架具有理论收敛保证，并且在经验上优于顺序后训练框架，同时具有相似的计算成本。

### 相关链接
* <a href="./papers/4034_Mitigating_Forgetting_in_.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=YeErX16hMC">ICLR链接</a>

### 摘要
Post-training of pre-trained LLMs, which typically consists of the supervised fine-tuning (SFT) stage and the preference learning (RLHF or DPO) stage, is crucial to effective and safe LLM applications. The widely adopted approach in post-training popular open-source LLMs is to sequentially perform SFT and RLHF/DPO. However, sequential training is sub-optimal in terms of SFT and RLHF/DPO trade-off: the LLM gradually forgets about the first stage's training when undergoing the second stage's training. We theoretically prove the sub-optimality of sequential post-training. Furthermore, we propose a practical joint post-training framework that has theoretical convergence guarantees and empirically outperforms sequential post-training framework, while having similar computational cost.
预训练的LLMs的后训练通常包括监督微调 (SFT) 阶段和偏好学习 (RLHF 或 DPO) 阶段，对于有效和安全的LLM申请至关重要。流行的开源LLMs培训后广泛采用的方法是依次执行 SFT 和 RLHF/DPO。然而，就 SFT 和 RLHF/DPO 权衡而言，顺序训练并不是最优的： LLM在进行第二阶段的训练时逐渐忘记了第一阶段的训练。我们从理论上证明了顺序后训练的次优性。此外，我们提出了一种实用的联合后训练框架，该框架具有理论收敛保证，并且在经验上优于顺序后训练框架，同时具有相似的计算成本。


## 33. CHiP: Cross-modal Hierarchical Direct Preference Optimization for Multimodal LLMs
多模式LLMs的跨模式分层DPO
### 关键字
* Multimodal Large Language Models
* Preference Optimization
* DPO
* Hallucination(幻觉)

### 主要内容
#### 针对问题：幻觉
作者对表征分布的分析表明，多模态 DPO 很难对齐图像和文本表征，并难以区分幻觉和非幻觉描述。
#### 提出CHiP：跨模式分层DPO
* 在 DPO 框架内引入了视觉偏好优化模块，使 MLLM 能够同时学习文本和视觉偏好
* 提出了一个分层文本偏好优化模块，该模块允许模型捕获多个粒度级别的偏好，包括response, segment和token级别。

### 相关链接
* <a href="./papers/7411_CHiP_Cross_modal_Hierarch.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=7lpDn2MhM2">ICLR链接</a>

### 摘要
Multimodal Large Language Models (MLLMs) still struggle with hallucinations despite their impressive capabilities. Recent studies have attempted to mitigate this by applying Direct Preference Optimization (DPO) to multimodal scenarios using preference pairs from text-based responses. However, our analysis of representation distributions reveals that multimodal DPO struggles to align image and text representations and to distinguish between hallucinated and non-hallucinated descriptions. To address these challenges, In this work, we propose a Cross-modal Hierarchical Direct Preference Optimization (CHiP) to address these limitations. We introduce a visual preference optimization module within the DPO framework, enabling MLLMs to learn from both textual and visual preferences simultaneously. Furthermore, we propose a hierarchical textual preference optimization module that allows the model to capture preferences at multiple granular levels, including response, segment, and token levels. We evaluate CHiP through both quantitative and qualitative analyses, with results across multiple benchmarks demonstrating its effectiveness in reducing hallucinations. On the Object HalBench dataset, CHiP outperforms DPO in hallucination reduction, achieving improvements of 52.7% and 55.5% relative points based on the base model Muffin and LLaVA models, respectively. We make all our datasets and code publicly available.
尽管多模态大语言模型（MLLM）具有令人印象深刻的能力，但仍然与幻觉作斗争。最近的研究试图通过使用基于文本的响应中的偏好对将直接偏好优化 (DPO) 应用于多模式场景来缓解这一问题。然而，我们对表征分布的分析表明，多模态 DPO 很难对齐图像和文本表征，并难以区分幻觉和非幻觉描述。为了应对这些挑战，在这项工作中，我们提出了跨模式分层直接偏好优化（CHiP）来解决这些限制。我们在 DPO 框架内引入了视觉偏好优化模块，使 MLLM 能够同时学习文本和视觉偏好。此外，我们提出了一个分层文本偏好优化模块，该模块允许模型捕获多个粒度级别的偏好，包括响应、分段和令牌级别。我们通过定量和定性分析来评估 CHiP，多个基准的结果证明了其在减少幻觉方面的有效性。在 Object HalBench 数据集上，CHiP 在减少幻觉方面优于 DPO，基于基础模型 Muffin 和 LLaVA 模型分别实现了 52.7% 和 55.5% 的相对点改进。我们公开所有数据集和代码。


## 34. Direct Alignment of Language Models via Quality-Aware Self-Refinement
通过质量意识自我完善直接调整语言模型
### 关键字
* RL
* Language Model
### 主要内容
#### DPO存在的问题
DPO 不考虑积极和消极反应的相对质量，并且可能导致次优的培训结果。
#### 解决方案
* 研究动态微调LLM中内在知识的使用，以获得相对质量并帮助细化损失函数。
    * 具体来说，利用LLM的知识来设计一个细化函数来估计正面和负面响应的质量
    * 表明构造的细化函数可以帮助在温和的假设下自细化损失函数。细化功能已集成到 DPO 及其变体身份策略优化 (IPO) 中。

### 相关链接
* <a href="./papers/4297_Direct_Alignment_of_Langu.pdfs">查看PDF</a>
* <a href="https://openreview.net/forum?id=tcdbBbHHPo">ICLR链接</a>

### 摘要
Reinforcement Learning from Human Feedback (RLHF) has been commonly used to align the behaviors of Large Language Models (LLMs) with human preferences. Recently, a popular alternative is Direct Policy Optimization (DPO), which replaces an LLM-based reward model with the policy itself, thus obviating the need for extra memory and training time to learn the reward model. However, DPO does not consider the relative qualities of the positive and negative responses, and can lead to sub-optimal training outcomes.
To alleviate this problem, we investigate the use of intrinsic knowledge within the on-the-fly fine-tuning LLM to obtain relative qualities and help to refine the loss function. Specifically, we leverage the knowledge of the LLM to design a refinement function to estimate the quality of both the positive and negative responses. We show that the constructed refinement function can help self-refine the loss function under mild assumptions. The refinement function is integrated into DPO and its variant Identity Policy Optimization (IPO).
Experiments across various evaluators indicate that they can improve the performance of the fine-tuned models over DPO and IPO.
基于人类反馈的强化学习 (RLHF) 通常用于使大型语言模型 ( LLMs ) 的行为与人类偏好保持一致。最近，一种流行的替代方案是直接策略优化（DPO），它用策略本身取代了基于LLM的奖励模型，从而无需额外的内存和训练时间来学习奖励模型。然而，DPO 不考虑积极和消极反应的相对质量，并且可能导致次优的培训结果。
为了缓解这个问题，我们研究了动态微调LLM中内在知识的使用，以获得相对质量并帮助细化损失函数。具体来说，我们利用LLM的知识来设计一个细化函数来估计正面和负面响应的质量。我们表明，构造的细化函数可以帮助在温和的假设下自细化损失函数。细化功能已集成到 DPO 及其变体身份策略优化 (IPO) 中。
不同评估者的实验表明，他们可以提高微调模型相对于 DPO 和 IPO 的性能。


## 35. Enhancing Multimodal LLM for Detailed and Accurate Video Captioning using Multi-Round Preference Optimization
使用`多轮偏好优化`增强多模态LLM以实现详细且准确的视频字幕
### 关键字
* Multi-modal LLM 
* Video Captioning
* Multi-Round DPO
* rebirth tuning(重生调优)
    * rebirth tuning可能是迭代再训练、模型复兴、解决过拟合和灾难性遗忘、引入新颖性和多样性？

### 主要内容


### 相关链接
* <a href="./papers/3578_Enhancing_Multimodal_LLM_.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=ufi0WPTgWp">ICLR链接</a>

### 摘要
Videos contain a wealth of information, and generating detailed and accurate descriptions in natural language is a key aspect of video understanding. In this paper, we present video-SALMONN 2, an advanced audio-visual large language model (LLM) with low-rank adaptation (LoRA) designed for enhanced video (with paired audio) captioning through directed preference optimization (DPO). We propose new metrics to evaluate the completeness and accuracy of video descriptions, which are optimized using DPO. To further improve training, we introduce a novel multi-round DPO (mrDPO) approach, which involves periodically updating the DPO reference model, merging and re-initializing the LoRA module as a proxy for parameter updates after each training round (1,000 steps), and incorporating guidance from ground-truth video captions to stabilize the process. To address potential catastrophic forgetting of non-captioning abilities due to mrDPO, we propose rebirth tuning, which finetunes the pre-DPO LLM by using the captions generated by the mrDPO-trained model as supervised labels. Experiments show that mrDPO significantly enhances video-SALMONN 2's captioning accuracy, reducing global and local error rates by 40% and 20%, respectively, while decreasing the repetition rate by 35%. The final video-SALMONN 2 model, with just 7 billion parameters, surpasses leading models such as GPT-4o and Gemini-1.5-Pro in video captioning tasks, while maintaining competitive performance to the state-of-the-art on widely used video question-answering benchmark among models of similar size. Upon acceptance, we will release the code, model checkpoints, and training and test data. Demos are available at https://video-salmonn-2.github.io.
视频包含丰富的信息，用自然语言生成详细而准确的描述是视频理解的一个关键方面。在本文中，我们提出了 video-SALMONN 2，这是一种具有低秩自适应 (LoRA) 的高级视听大语言模型 ( LLM )，旨在通过定向偏好优化 (DPO) 增强视频（带有配对音频）字幕。我们提出了新的指标来评估视频描述的完整性和准确性，并使用 DPO 进行了优化。为了进一步改进训练，我们引入了一种新颖的多轮 DPO (mrDPO) 方法，该方法涉及定期更新 DPO 参考模型、合并并重新初始化 LoRA 模块作为每轮训练（1,000 步）后参数更新的代理，并结合真实视频字幕的指导来稳定该过程。为了解决由于 mrDPO 导致的非字幕能力的潜在灾难性遗忘，我们提出了重生调整，即通过使用 mrDPO 训练模型生成的字幕作为监督标签来微调预 DPO LLM 。实验表明，mrDPO 显着提高了 video-SALMONN 2 的字幕准确性，将全局和局部错误率分别降低了 40% 和 20%，同时将重复率降低了 35%。最终的视频 SALMONN 2 模型仅具有 70 亿个参数，在视频字幕任务中超越了 GPT-4o 和 Gemini-1.5-Pro 等领先模型，同时在广泛使用的视频上保持了与最先进的竞争性能类似尺寸模型中的问答基准。接受后，我们将发布代码、模型检查点以及训练和测试数据。演示可在https://video-salmonn-2.github.io获取。



## 
### 关键字


### 主要内容


### 相关链接
* <a href="">查看PDF</a>
* <a href="">ICLR链接</a>

### 摘要



## 
### 关键字


### 主要内容


### 相关链接
* <a href="">查看PDF</a>
* <a href="">ICLR链接</a>

### 摘要



## 
### 关键字


### 主要内容


### 相关链接
* <a href="">查看PDF</a>
* <a href="">ICLR链接</a>

### 摘要



## 
### 关键字


### 主要内容


### 相关链接
* <a href="">查看PDF</a>
* <a href="">ICLR链接</a>

### 摘要



## 
### 关键字


### 主要内容


### 相关链接
* <a href="">查看PDF</a>
* <a href="">ICLR链接</a>

### 摘要



## 
### 关键字


### 主要内容


### 相关链接
* <a href="">查看PDF</a>
* <a href="">ICLR链接</a>

### 摘要



## 
### 关键字


### 主要内容


### 相关链接
* <a href="">查看PDF</a>
* <a href="">ICLR链接</a>

### 摘要



## 
### 关键字


### 主要内容


### 相关链接
* <a href="">查看PDF</a>
* <a href="">ICLR链接</a>

### 摘要



## 
### 关键字


### 主要内容


### 相关链接
* <a href="">查看PDF</a>
* <a href="">ICLR链接</a>

### 摘要



## 
### 关键字


### 主要内容


### 相关链接
* <a href="">查看PDF</a>
* <a href="">ICLR链接</a>

### 摘要



## 
### 关键字


### 主要内容


### 相关链接
* <a href="">查看PDF</a>
* <a href="">ICLR链接</a>

### 摘要



## 
### 关键字


### 主要内容


### 相关链接
* <a href="">查看PDF</a>
* <a href="">ICLR链接</a>

### 摘要