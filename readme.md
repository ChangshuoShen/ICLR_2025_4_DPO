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
RLHF （RLHF） 使大型语言模型 （LLMs人类偏好保持一致。然而，由于外部因素（例如环境变化和社会影响），这些偏好通常会随着时间的推移而改变。因此，当时的错误可能现在就是正确的。当前的偏好优化算法在其建模中没有考虑时间偏好漂移，这可能导致严重的错位。为了解决这一限制，我们使用了动态 Bradley-Terry 模型，该模型通过瞬态奖励函数对偏好进行建模，并提出了非平稳直接偏好优化 （NS-DPO）。通过在损失函数中引入 discount 参数，NS-DPO 应用指数加权，按比例将学习集中在与时间相关的更多数据点上。我们从理论上分析了 NS-DPO 在离线设置中的收敛性，提供了由非平稳偏好引起的估计误差的上限。最后，我们证明了 NS-DPO1 在具有漂移偏好的情况下微调 LLMs。通过使用著名的奖励模型模拟偏好漂移并相应地修改流行的 LLM 数据集，我们表明 NS-DPO 微调LLMs 在非平稳性下保持稳健性，明显优于忽略时间偏好变化的基线算法，而不会牺牲平稳情况下的性能。

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
RLHF （RLHF） 和直接偏好优化 （DPO） 是新兴的重要技术，用于将大型语言模型 （LLM人类偏好保持一致。然而，RLHF 和 DPO 训练的质量受到 C或偏好、奖励 O验证和对 Verbosity 的偏向的严重影响。据我们所知，现有的大多数工作只解决了这些重要问题中的一个，而其他少数工作需要大量的计算来估计多个奖励模型，缺乏泛化能力的理论保证。在这项工作中，我们提出了 RLHF-COV 和 DPO-COV 算法，它们可以在离线和在线环境中同时缓解这三个问题。这种能力可以通过为在损坏数据上训练的 DPO-COV 算法获得长度正则化泛化错误率来理论上证明，这与具有干净数据且没有长度正则化的简单情况的最已知率相匹配。此外，我们的 DPO-COV 算法无需奖励估计即可轻松实现，并被证明等同于我们的 RLHF-COV 算法，这直接暗示了原版 RLHF 和 DPO 算法之间的等效性。

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
LLM 在数万亿个token上进行了预训练，但预训练的 LLM 可能仍会生成不需要的响应。为了解决这个问题，提出了 RLHF 、 DPO 和 KTO 等对准技术。但是，这些对齐技术具有局限性。例如，RLHF 需要分别训练奖励模型和策略，训练过程中复杂、耗时、内存密集且不稳定。DPO 提出了最优策略和奖励之间的映射，大大简化了 RLHF 的训练过程。但是，它不能充分利用奖励模型，并且仅限于成对偏好数据。
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
* 提出建议DPO的最佳数据对于获胜和失败响应中的每个token有相同的预期奖励，因为token重要性没有差异
* 但是，因为实际中无法获得最优数据集，所以建议使用原始数据进行重要性采样，以实现无偏优化
#### TIS(Token-level Importance Sampling)-DPO
根据每个token的奖励为其分配重要度权重

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
直接偏好优化（DPO）是一种广泛采用的离线偏好优化算法，旨在使用成对偏好数据将大型语言模型（ LLMs ）与人类期望的行为结合起来。然而，成对数据中的获胜响应和失败响应是单独生成的，导致它们之间的相关性较弱以及对齐性能不佳。为了解决这个问题，我们提出了一种有效的成对数据桥接和建模相关性框架，名为 BMC。首先，我们通过有针对性的修改来提高成对偏好信号的一致性和信息量，以获胜响应为参考，通过改进失败响应来合成伪获胜响应。其次，我们发现仅 DPO 不足以对这些相关性进行建模并捕获细微的变化。因此，我们建议通过在训练期间动态利用策略模型的置信度来学习token级相关性。关于 QA、数学和指令遵循任务的综合实验证明了我们方法的有效性，显着超越了包括 DPO 在内的竞争基准。此外，我们深入的定量分析揭示了我们的方法优于 DPO 的原因，并展示了其与其他 DPO 变体的多功能性。

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
尽管多模态大语言模型（MLLM）具有令人印象深刻的能力，但仍然与幻觉作斗争。最近的研究试图通过使用基于文本的响应中的偏好对将直接偏好优化 (DPO) 应用于多模式场景来缓解这一问题。然而，我们对表征分布的分析表明，多模态 DPO 很难对齐图像和文本表征，并难以区分幻觉和非幻觉描述。为了应对这些挑战，在这项工作中，我们提出了跨模式分层直接偏好优化（CHiP）来解决这些限制。我们在 DPO 框架内引入了视觉偏好优化模块，使 MLLM 能够同时学习文本和视觉偏好。此外，我们提出了一个分层文本偏好优化模块，该模块允许模型捕获多个粒度级别的偏好，包括响应、分段和token级别。我们通过定量和定性分析来评估 CHiP，多个基准的结果证明了其在减少幻觉方面的有效性。在 Object HalBench 数据集上，CHiP 在减少幻觉方面优于 DPO，基于基础模型 Muffin 和 LLaVA 模型分别实现了 52.7% 和 55.5% 的相对点改进。我们公开所有数据集和代码。


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
#### 提出video-SALMONN2
* 一种具有低秩自适应 (LoRA, low rank adaptation) 的高级视听大语言模型 ( LLM )，旨在通过定向偏好优化 (DPO) 增强视频（带有配对音频）字幕。
* 提出新的指标来评估视频描述的完整性和准确性，并使用 DPO 进行了优化。
#### 改进训练mrDPO(multi round)
* 涉及定期更新 DPO 参考模型、合并并重新初始化 LoRA 模块作为每轮训练（1,000 步）后参数更新的代理，并结合真实视频字幕的指导来稳定该过程。
* 为了解决由于 mrDPO 导致的非字幕能力的潜在灾难性遗忘，提出了rebirth tuning(重生调整)，即通过使用 mrDPO 训练模型生成的字幕作为监督标签来微调预 DPO LLM 。
### 相关链接
* <a href="./papers/3578_Enhancing_Multimodal_LLM_.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=ufi0WPTgWp">ICLR链接</a>

### 摘要
Videos contain a wealth of information, and generating detailed and accurate descriptions in natural language is a key aspect of video understanding. In this paper, we present video-SALMONN 2, an advanced audio-visual large language model (LLM) with low-rank adaptation (LoRA) designed for enhanced video (with paired audio) captioning through directed preference optimization (DPO). We propose new metrics to evaluate the completeness and accuracy of video descriptions, which are optimized using DPO. To further improve training, we introduce a novel multi-round DPO (mrDPO) approach, which involves periodically updating the DPO reference model, merging and re-initializing the LoRA module as a proxy for parameter updates after each training round (1,000 steps), and incorporating guidance from ground-truth video captions to stabilize the process. To address potential catastrophic forgetting of non-captioning abilities due to mrDPO, we propose rebirth tuning, which finetunes the pre-DPO LLM by using the captions generated by the mrDPO-trained model as supervised labels. Experiments show that mrDPO significantly enhances video-SALMONN 2's captioning accuracy, reducing global and local error rates by 40% and 20%, respectively, while decreasing the repetition rate by 35%. The final video-SALMONN 2 model, with just 7 billion parameters, surpasses leading models such as GPT-4o and Gemini-1.5-Pro in video captioning tasks, while maintaining competitive performance to the state-of-the-art on widely used video question-answering benchmark among models of similar size. Upon acceptance, we will release the code, model checkpoints, and training and test data. Demos are available at https://video-salmonn-2.github.io.
视频包含丰富的信息，用自然语言生成详细而准确的描述是视频理解的一个关键方面。在本文中，我们提出了 video-SALMONN 2，这是一种具有低秩自适应 (LoRA) 的高级视听大语言模型 ( LLM )，旨在通过定向偏好优化 (DPO) 增强视频（带有配对音频）字幕。我们提出了新的指标来评估视频描述的完整性和准确性，并使用 DPO 进行了优化。为了进一步改进训练，我们引入了一种新颖的多轮 DPO (mrDPO) 方法，该方法涉及定期更新 DPO 参考模型、合并并重新初始化 LoRA 模块作为每轮训练（1,000 步）后参数更新的代理，并结合真实视频字幕的指导来稳定该过程。为了解决由于 mrDPO 导致的非字幕能力的潜在灾难性遗忘，我们提出了重生调整，即通过使用 mrDPO 训练模型生成的字幕作为监督标签来微调预 DPO LLM 。实验表明，mrDPO 显着提高了 video-SALMONN 2 的字幕准确性，将全局和局部错误率分别降低了 40% 和 20%，同时将重复率降低了 35%。最终的视频 SALMONN 2 模型仅具有 70 亿个参数，在视频字幕任务中超越了 GPT-4o 和 Gemini-1.5-Pro 等领先模型，同时在广泛使用的视频上保持了与最先进的竞争性能类似尺寸模型中的问答基准。接受后，我们将发布代码、模型检查点以及训练和测试数据。演示可在https://video-salmonn-2.github.io获取。

## 36. Scalable Ranked Preference Optimization for Text-to-Image Generation
用于文生图的可扩展排名偏好优化
### 关键字
* Text2Image Generation
* DPO
* Learning from AI Feedback

### 主要内容
#### DPO用在T2I领域中的问题
* 标记大规模数据集消耗很多资源
* T2I模型更新速度快，带来更高质量的图像的时候人类偏好数据集会很快过时
#### 引入RankDPO
* 研究了一种可扩展的方法，用于收集用于 DPO 训练的大规模且完全合成的数据集。
    * 具体来说，对配对图像的偏好是使用预先训练的奖励函数生成的，消除了人类参与注释过程的需要，极大地提高了数据集收集效率。
    * 证明此类数据集允许对多个模型进行平均预测并收集排名偏好，而不是成对偏好。
* 引入 RankDPO 来使用排名反馈增强基于 DPO 的方法。
### 相关链接
* <a href="./papers/7828_Scalable_Ranked_Preferenc.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=Y6KUBkUimC">ICLR链接</a>

### 摘要
Direct Preference Optimization (DPO) has emerged as a powerful approach to align text-to-image (T2I) models with human feedback. Unfortunately, successful application of DPO to T2I models requires a huge amount of resources to collect and label large-scale datasets, e.g., millions of generated paired images annotated with human preferences. In addition, these human preference datasets can get outdated quickly as the rapid improvements of T2I models lead to higher quality images. In this work, we investigate a scalable approach for collecting large-scale and fully synthetic datasets for DPO training. Specifically, the preferences for paired images are generated using a pre-trained reward function, eliminating the need for involving humans in the annotation process, greatly improving the dataset collection efficiency. Moreover, we demonstrate that such datasets allow averaging predictions across multiple models and collecting ranked preferences as opposed to pairwise preferences. Furthermore, we introduce RankDPO to enhance DPO-based methods using the ranking feedback. Applying RankDPO on SDXL and SD3-Medium models with our synthetically generated preference dataset ``Syn-Pic'' improves both prompt-following (on benchmarks like T2I-Compbench, GenEval, and DPG-Bench) and visual quality (through user studies). This pipeline presents a practical and scalable solution to develop better preference datasets to enhance the performance and safety of text-to-image models.
直接偏好优化 (DPO) 已成为一种将文本到图像 (T2I) 模型与人类反馈结合起来的强大方法。不幸的是，DPO 成功应用于 T2I 模型需要大量资源来收集和标记大规模数据集，例如，生成的数百万张带有人类偏好注释的配对图像。此外，随着 T2I 模型的快速改进带来更高质量的图像，这些人类偏好数据集可能很快就会过时。在这项工作中，我们研究了一种可扩展的方法，用于收集用于 DPO 训练的大规模且完全合成的数据集。具体来说，对配对图像的偏好是使用预先训练的奖励函数生成的，消除了人类参与注释过程的需要，极大地提高了数据集收集效率。此外，我们证明此类数据集允许对多个模型进行平均预测并收集排名偏好，而不是成对偏好。此外，我们引入 RankDPO 来使用排名反馈增强基于 DPO 的方法。将 RankDPO 应用于 SDXL 和 SD3-Medium 模型以及我们综合生成的偏好数据集“Syn-Pic”，可以提高提示跟踪（在 T2I-Compbench、GenEval 和 DPG-Bench 等基准上）和视觉质量（通过用户研究） 。该管道提供了一种实用且可扩展的解决方案，用于开发更好的偏好数据集，以增强文本到图像模型的性能和安全性。


## 37. Simultaneous Reward Distillation and Preference Learning: Get You a Language Model Who Can Do Both
同时进行奖励蒸馏和偏好学习：为您提供一个可以同时完成这两项任务的语言模型
* 蒸馏学习：将大型、复杂的模型压缩为较小的模型，以便在不显著降低性能的情况下提高计算效率。该方法通过“教师-学生”框架实现，教师模型在训练过程中会生成软标签或概率分布，将知识传递给学生模型，使其能够在较少资源下实现接近教师模型的效果。
### 关键字
* Preference Optimization
* Reward Distillation(奖励蒸馏)
* LLMs

### 主要内容
#### DPO和类似的直接对齐方法的问题
* 可能导致退化策略
* 并且严重依赖基于 Bradley-Terry 的偏好公式来对候选输出对之间的奖励差异进行建模。该公式受到非确定性或噪声偏好标签的挑战，
#### DRDO(Direct Reward Distillation and policy-Optimization)
* 这是一种基于监督知识蒸馏的偏好调整方法，可以同时对奖励和偏好进行建模，以避免这种退化。 
* DRDO 直接模仿预言机分配的奖励，同时从新颖的偏好可能性公式中学习人类偏好。
### 相关链接
* <a href="./papers/8023_Simultaneous_Reward_Disti.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=l9LWx9HMl5">ICLR链接</a>

### 摘要
Reward modeling of human preferences is one of the cornerstones of building usable generative large language models (LLMs). While traditional RLHF-based alignment methods explicitly maximize the expected rewards from a separate reward model, more recent supervised alignment methods like Direct Preference Optimization (DPO) circumvent this phase to avoid problems including model drift and reward overfitting. Although popular due to its simplicity, DPO and similar direct alignment methods can still lead to degenerate policies, and rely heavily on the Bradley-Terry-based preference formulation to model reward differences between pairs of candidate outputs. This formulation is challenged by non-deterministic or noisy preference labels, for example human scoring of two candidate outputs is of low confidence. In this paper, we introduce DRDO (Direct Reward Distillation and policy-Optimization), a supervised knowledge distillation-based preference alignment method that simultaneously models rewards and preferences to avoid such degeneracy. DRDO directly mimics rewards assigned by an oracle while learning human preferences from a novel preference likelihood formulation. Our experimental results on the Ultrafeedback and TL;DR datasets demonstrate that policies trained using DRDO surpass previous methods such as DPO and e-DPO in terms of expected rewards and are more robust, on average, to noisy preference signals as well as out-of-distribution (OOD) settings.
人类偏好的奖励建模是构建可用的生成大语言模型（ LLMs ）的基石之一。虽然传统的基于 RLHF 的对齐方法明确地最大化了单独奖励模型的预期奖励，但最近的监督对齐方法（例如直接偏好优化（DPO））绕过了此阶段，以避免模型漂移和奖励过度拟合等问题。尽管因其简单性而广受欢迎，但 DPO 和类似的直接对齐方法仍然可能导致退化策略，并且严重依赖基于 Bradley-Terry 的偏好公式来对候选输出对之间的奖励差异进行建模。该公式受到非确定性或噪声偏好标签的挑战，例如两个候选输出的人工评分置信度较低。在本文中，我们介绍了 DRDO（直接奖励蒸馏和策略优化），这是一种基于监督知识蒸馏的偏好调整方法，可以同时对奖励和偏好进行建模，以避免这种退化。 DRDO 直接模仿预言机分配的奖励，同时从新颖的偏好可能性公式中学习人类偏好。我们在 Ultrafeedback 和 TL;DR 数据集上的实验结果表明，使用 DRDO 训练的策略在预期奖励方面超越了以前的方法，例如 DPO 和 e-DPO，并且平均而言，对于噪声偏好信号以及非- 分发 (OOD) 设置。

## 38. Improving Inverse Folding for Peptide Design with Diversity-Regularized Direct Preference Optimization
通过多样性正则化DPO改进肽设计的反向折叠
### 关键字
* Inverse Folding
* Structure-based design
* Peptide Design(胚设计): 优化多肽序列以获得特定的三维结构
* DPO
### 主要内容
#### ProteinMPNN问题
这些模型很容易生成不折叠到参考结构中的重复序列
#### 使用DPO微调ProteinMPNN
* 产生多样化且结构一致的肽序列
* 两项增强
    * 在线多样性正则化
    * 特定领域先验
### 相关链接
* <a href="./papers/11912_Improving_Inverse_Foldin.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=VY96NfQRIo">ICLR链接</a>

### 摘要
Inverse folding models play an important role in structure-based design by predicting amino acid sequences that fold into desired reference structures. Models like ProteinMPNN, a message-passing encoder-decoder model, are trained to reliably produce new sequences from a reference structure. However, when applied to peptides, these models are prone to generating repetitive sequences that do not fold into the reference structure. To address this, we finetune ProteinMPNN to produce diverse and structurally consistent peptide sequences via Direct Preference Optimization (DPO). We derive two enhancements to DPO: online diversity regularization and domain-specific priors.Additionally, we develop a new understanding on improving diversity in decoder models. When conditioned on OpenFold generated structures, our finetuned models achieve state-of-the-art structural similarity scores, improving base ProteinMPNN by at least 8%. Compared to standard DPO, our regularized method achieves up to 20% higher sequence diversity with no loss in structural similarity score.
反向折叠模型通过预测折叠成所需参考结构的氨基酸序列，在基于结构的设计中发挥着重要作用。 ProteinMPNN（一种消息传递编码器-解码器模型）等模型经过训练，可以从参考结构可靠地生成新序列。然而，当应用于肽时，这些模型很容易生成不折叠到参考结构中的重复序列。为了解决这个问题，我们通过直接偏好优化（DPO）对 ProteinMPNN 进行微调，以产生多样化且结构一致的肽序列。我们对 DPO 进行了两项增强：在线多样性正则化和特定领域先验。此外，我们对提高解码器模型的多样性有了新的理解。当以 OpenFold 生成的结构为条件时，我们的微调模型实现了最先进的结构相似性评分，将基础 ProteinMPNN 提高了至少 8%。与标准 DPO 相比，我们的正则化方法实现了高达 20% 的序列多样性提高，并且结构相似性得分没有损失。





## 39. *Exploratory Preference Optimization: Provably Sample-Efficient Exploration in RLHF with General Function Approximation
探索性偏好优化：使用一般函数逼近在 RLHF 中进行可证明的样本有效探索 
### 关键字
* Learning Theory
* RL Theory
* Sample-Efficient RL 
### 主要内容
#### 探索一个基本问题：
如何在偏好反馈和通用函数逼近下以online方式进行高效探索
#### XPO(Exploratory Preference Optimization)
* 只需要对(在线)DPO进行一行修改，但却提供了已知的最强的可证明保证
* XPO 通过新颖且有原则的探索奖励增强了 DPO 目标，使算法能够在初始模型和偏好反馈数据的支持之外进行战略性探索。
* 证明无论初始模型的覆盖范围如何，XPO 都具有可证明的样本效率，并且在自然探索条件下收敛到接近最优的策略。
### 相关链接
* <a href="./papers/8265_Exploratory_Preference_Op.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=QYigQ6gXNw">ICLR链接</a>

### 摘要
This paper investigates a basic question in reinforcement learning from human feedback (RLHF) from a theoretical perspective: how to efficiently explore in an online manner under preference feedback and general function approximation. We take the initial step towards a theoretical understanding of this problem by proposing a novel algorithm, Exploratory Preference Optimization (XPO). This algorithm is elegantly simple---requiring only a one-line modification to (online) Direct Preference Optimization (DPO; Rafailov et al., 2023)---yet provides the strongest known provable guarantees. XPO augments the DPO objective with a novel and principled exploration bonus, enabling the algorithm to strategically explore beyond the support of the initial model and preference feedback data. We prove that XPO is provably sample-efficient and converges to a near-optimal policy under natural exploration conditions, regardless of the initial model's coverage. Our analysis builds on the observation that DPO implicitly performs a form of Bellman error minimization. It synthesizes previously disparate techniques from language modeling and theoretical reinforcement learning in a serendipitous fashion through the lens of KL-regularized Markov decision processes.
本文从理论角度研究了人类反馈强化学习（RLHF）的一个基本问题：如何在偏好反馈和通用函数逼近下以在线方式进行高效探索。我们通过提出一种新颖的算法——探索性偏好优化（XPO），迈出了对该问题的理论理解的第一步。该算法非常简单——只需要对（在线）直接偏好优化进行一行修改（DPO；Rafailov 等人，2023）——但却提供了已知的最强的可证明保证。 XPO 通过新颖且有原则的探索奖励增强了 DPO 目标，使算法能够在初始模型和偏好反馈数据的支持之外进行战略性探索。我们证明，无论初始模型的覆盖范围如何，XPO 都具有可证明的样本效率，并且在自然探索条件下收敛到接近最优的策略。我们的分析建立在 DPO 隐式执行贝尔曼误差最小化形式的观察之上。它通过KL 正则化马尔可夫决策过程的镜头，以偶然的方式综合了以前来自语言建模和理论强化学习的不同技术。

## 40. AIPO: Agreement-Aware Iterative Preference Optimization for Length Exploitation Mitigation
AIPO：用于缓解长度利用的协议感知迭代偏好优化  
### 关键字
* LM
* Alignment
### 主要内容
#### DPO长度利用问题
DPO 中的长度利用问题变得越来越严重在迭代偏好优化期间更加明显，每次迭代的严重性逐渐升级。
#### 此工作结果
* 介绍在构建迭代偏好优化管道时的发现和分析
* 分析迭代过程中的长度利用问题，并提出了一种用于迭代偏好优化的新训练目标，即AIPO(Agreement-Aware Iterative Preference Optimization)
### 相关链接
* <a href="./papers/6020_AIPO_Agreement_Aware_Iter.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=ixdAVqjShn">ICLR链接</a>

### 摘要
Direct Preference Optimization (DPO) is gaining popularity as an alternative to Proximal Policy Optimization (PPO) for aligning Large Language Models (LLMs). Recent research on aligning LLMs iteratively with synthetic or partially synthetic data has shown promising outcomes, facilitating the scalability of DPO training in both academic settings and proprietary models such as Llama 3. Despite its success, we observe that the issue of length exploitation in DPO becomes more pronounced during iterative preference optimization, with the severity escalating progressively with each iteration. This observation prompts an in-depth examination of iterative preference optimization with synthetic data. In this paper, we present our findings and analyses in building our iterative preference optimization pipeline. Specifically, we analyze the issue of length exploitation in this iterative process and propose a novel training objective for iterative preference optimization, namely \textbf{A}greement-aware \textbf{I}terative \textbf{P}reference \textbf{O}ptimization (AIPO). To demonstrate the effectiveness of our proposed method, we conduct extensive experiments and show that it achieves state-of-the-art performance on MT-Bench, AlpacaEval 2.0, and Arena-Hard.
直接偏好优化 (DPO) 作为用于对齐大型语言模型 ( LLMs ) 的邻近策略优化 (PPO) 的替代方案越来越受欢迎。最近关于将LLMs与合成或部分合成数据进行迭代调整的研究显示出有希望的结果，促进了学术环境和 Llama 3 等专有模型中 DPO 培训的可扩展性。尽管取得了成功，但我们观察到 DPO 中的长度利用问题变得越来越严重在迭代偏好优化期间更加明显，每次迭代的严重性逐渐升级。这一观察结果促使人们对合成数据的迭代偏好优化进行深入研究。在本文中，我们介绍了在构建迭代偏好优化管道时的发现和分析。具体来说，我们分析了迭代过程中的长度利用问题，并提出了一种用于迭代偏好优化的新训练目标，即 \textbf{A}greement-aware \textbf{I}terative \textbf{P}reference \textbf{O}优化（AIPO）。为了证明我们提出的方法的有效性，我们进行了广泛的实验，并表明它在 MT-Bench、AlpacaEval 2.0 和 Arena-Hard 上实现了最先进的性能。


## 41. TPO: Aligning Large Language Models with Multi-branch & Multi-step Preference Trees
TPO：将大型语言模型与多分支和多步骤偏好树对齐
### 关键字
* RLHF
* LM
* Preference Learning
### 主要内容
#### DPO在推理领域的局限
基于二元偏好优化的DPO算法无法学习偏好树提供的不同偏好/不偏好程度的多个响应，导致偏好学习不完整。
#### TPO(Tree PO)
它不会从偏好树中采样配对偏好响应；相反，它在微调过程中直接从整个偏好树中学习。
* TPO 将语言模型对齐制定为偏好列表排名问题，其中策略可以根据给定提示从排名的响应偏好列表中更有效地学习。
* 此外，为了进一步帮助LLMs识别长链推理中的判别步骤并增加偏好列表中的相对奖励裕度，TPO利用自适应步骤奖励来调整轨迹中每个步骤的奖励值，以进行细粒度的偏好优化。

### 相关链接
* <a href="./papers/9326_TPO_Aligning_Large_Langua.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=O0sQ9CPzai">ICLR链接</a>

### 摘要



## 42. A Tailored Framework for Aligning Diffusion Models with Human Preference
使扩散模型与人类偏好保持一致的定制框架
### 关键字
* RLHF
* Diffusion Model
* DPO
### 主要内容
#### DPO在T2I领域中的一致性假设存在问题
* 以前的方法通常在中间步骤中假设最终生成的图像与其相应的噪声样本之间具有一致的偏好标签，并直接将 DPO 应用于这些噪声样本进行微调
* 但是根据最终偏好顺序直接将 DPO 应用于来自不同生成轨迹的噪声样本可能会破坏优化过程。
#### TailorPO框架
1. 从梯度方向和偏好顺序两个角度论证了先前方法中固有的问题
2. 提出了一个定制PO(TailorPO) 框架，用于使扩散模型与人类偏好保持一致，并以一些理论见解为基础。
3. 根据中间噪声样本的逐步奖励直接对它们的偏好顺序进行排序，并通过简单而高效的设计有效解决优化方向问题。
4. （第一个考虑扩散模型的独特结构并利用偏好对齐中的梯度引导来增强优化效果的人。

### 相关链接
* <a href="./papers/1717_A_Tailored_Framework_for_.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=9LZna4ryFH">ICLR链接</a>

### 摘要
The direct preference optimization (DPO) method has shown success in aligning text-to-image diffusion models with human preference. Previous approaches typically assume a consistent preference label between final generated images and their corresponding noisy samples at intermediate steps, and directly apply DPO to these noisy samples for fine-tuning. However, we identify a significant issue with this consistency assumption, as directly applying DPO to noisy samples from different generation trajectories based on final preference order may disrupt the optimization process. We first demonstrate the issues inherent in previous methods from two perspectives: gradient direction and preference order, and then propose a Tailored Preference Optimization (TailorPO) framework for aligning diffusion models with human preference, underpinned by some theoretical insights. Our approach directly ranks the preference order of intermediate noisy samples based on their step-wise reward, and effectively resolves the optimization direction issues through a simple yet efficient design. Additionally, to the best of our knowledge, we are the first to consider the distinct structure of diffusion models and leverage the gradient guidance in preference aligning to enhance the optimization effectiveness. Experimental results demonstrate that our method significantly improves the model's ability to generate aesthetically pleasing and human-preferred images.
直接偏好优化（DPO）方法已成功地将文本到图像的扩散模型与人类偏好结合起来。以前的方法通常在中间步骤中假设最终生成的图像与其相应的噪声样本之间具有一致的偏好标签，并直接将 DPO 应用于这些噪声样本进行微调。然而，我们发现这种一致性假设存在一个重大问题，因为根据最终偏好顺序直接将 DPO 应用于来自不同生成轨迹的噪声样本可能会破坏优化过程。我们首先从梯度方向和偏好顺序两个角度论证了先前方法中固有的问题，然后提出了一个定制的P参考优化(TailorPO) 框架，用于使扩散模型与人类偏好保持一致，并以一些理论见解为基础。我们的方法根据中间噪声样本的逐步奖励直接对它们的偏好顺序进行排序，并通过简单而高效的设计有效解决优化方向问题。此外，据我们所知，我们是第一个考虑扩散模型的独特结构并利用偏好对齐中的梯度引导来增强优化效果的人。实验结果表明，我们的方法显着提高了模型生成美观且人类喜欢的图像的能力。


## 43. DIPPER: Direct Preference Optimization for Primitive-Enabled Hierarchical Reinforcement Learning
DIPPER：基于原语的分层强化学习的直接偏好优化 (原语：即基础的操作或子策略)
### 关键字
* Hierarchical RL (分层强化学习)
* Preference Learning

### 主要内容
#### 多层次并发学习触发问题
* 由于较低级别原语的非平稳行为，多个层次上的并发学习策略经常会受到训练不稳定的影响
* 直接将 DPO 应用于 HRL 的更高级别是无效的，并且会导致不可行的子目标生成问题
#### DIPPER
* 一种高效的分层框架，它利用DPO来减轻较高级别的非平稳性，同时使用强化学习来训练较低级别的相应基元。
* 开发了一个新颖的、基于上层政策学习的下层原始正则化的原则框架。利用双层优化为所提出的框架提供了理论依据。

### 相关链接
* <a href="./papers/5077_DIPPER_Direct_Preference_.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=mJKhn7Ey4y">ICLR链接</a>

### 摘要
Hierarchical reinforcement learning (HRL) is an elegant framework for learning efficient control policies to perform complex robotic tasks, especially in sparse reward settings. However, concurrently learning policies at multiple hierarchical levels often suffers from training instability due to non-stationary behavior of lower-level primitives. In this work, we introduce DIPPER, an efficient hierarchical framework that leverages Direct Preference Optimization (DPO) to mitigate non-stationarity at the higher level, while using reinforcement learning to train the corresponding primitives at the lower level. We observe that directly applying DPO to the higher level in HRL is ineffective and leads to infeasible subgoal generation issues. To address this, we develop a novel, principled framework based on lower-level primitive regularization of upper-level policy learning. We provide a theoretical justification for the proposed framework utilizing bi-level optimization. The application of DPO also necessitates the development of a novel reference policy formulation for feasible subgoal generation. To validate our approach, we conduct extensive experimental analyses on a variety of challenging, sparse-reward robotic navigation and manipulation tasks. Our results demonstrate that DIPPER shows impressive performance and demonstrates an improvement of up to 40% over the baselines in complex sparse robotic control tasks.
分层强化学习（HRL）是一个优雅的框架，用于学习有效的控制策略来执行复杂的机器人任务，特别是在稀疏奖励设置中。然而，由于较低级别原语的非平稳行为，多个层次上的并发学习策略经常会受到训练不稳定的影响。在这项工作中，我们引入了 DIPPER，这是一种高效的分层框架，它利用直接偏好优化（DPO）来减轻较高级别的非平稳性，同时使用强化学习来训练较低级别的相应基元。我们观察到，直接将 DPO 应用于 HRL 的更高级别是无效的，并且会导致不可行的子目标生成问题。为了解决这个问题，我们开发了一个新颖的、基于上层政策学习的下层原始正则化的原则框架。我们利用双层优化为所提出的框架提供了理论依据。 DPO 的应用还需要开发一种新颖的参考策略制定来生成可行的子目标。为了验证我们的方法，我们对各种具有挑战性的、稀疏奖励的机器人导航和操作任务进行了广泛的实验分析。我们的结果表明，DIPPER 显示了令人印象深刻的性能，并且在复杂的稀疏机器人控制任务中比基线提高了高达 40%。

## 44. 3D-CT-GPT++: Enhancing 3D Radiology Report Generation with Direct Preference Optimization and Large Vision-Language Models
3D-CT-GPT++：通过DPO和大型视觉语言模型(LVLM)增强 3D 放射学报告生成  
### 关键字
* Radiology Report Generation
* 3D Medical Imaging
* DPO
* MLLMs

### 主要内容
#### 当前生成3D报告的问题
* 当前生成 3D 报告的方法通常采用视频处理方法，该方法很难有效捕获沿 Z 轴的关系。
* 此外，用于生成 3D 图像报告的基于多模态大语言模型的方法面临着重大限制，特别是在图像编码器表示 3D 结构和生成内容中出现的幻觉的能力方面。
#### 提出3D-CT-GPT++
* 该模型集成了优化的 3D 图像编码器 CTViT-V，专为胸部CT扫描设计，并建立在 LLaVA-1.5 架构之上
* 使用DPO，利用GPT-4对SFT模型的输出做评分，为后续DPO创建偏好数据集

### 相关链接
* <a href="./papers/13584_3D_CT_GPT_Enhancing_3D_R.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=LzycEbgLoi">ICLR链接</a>

### 摘要
Automatically generating radiology reports from three-dimensional medical images, such as 3D CT scans, plays a crucial role in modern diagnostics. Current approaches for generating 3D reports often adopt video processing methods, which struggle to effectively capture the relationships along the Z-axis. Additionally, multimodal large language model-based methods for generating 3D image reports face significant limitations, particularly in terms of the image encoder’s ability to represent 3D structures and the hallucinations that arise in generated content. To address these challenges, we propose the 3D-CT-GPT++ model. This model integrates the optimized 3D image encoder CTViT-V, specifically designed for chest CT scans, and builds upon the LLaVA-1.5 architecture. Furthermore, we introduce \textit{Direct Preference Optimization (DPO)}, where GPT-4 is used to score the outputs of our fully fine-tuned (SFT) model, creating a preference dataset for subsequent DPO training. DPO significantly reduces hallucinations in the report generation process, ensuring the generated reports are more aligned with clinical needs. We fine-tuned the model on both high-quality private and public datasets to ensure clinical relevance. Extensive experiments were conducted using standard natural language generation (NLG) evaluation metrics, including BLEU, METEOR, and ROUGE-L, to assess the report generation performance. Experimental results demonstrate that 3D-CT-GPT++ significantly outperforms existing methods in terms of accuracy, fluency, and clinical relevance, advancing the automation of 3D medical report generation.
从三维医学图像（例如 3D CT 扫描）自动生成放射学报告在现代诊断中发挥着至关重要的作用。当前生成 3D 报告的方法通常采用视频处理方法，该方法很难有效捕获沿 Z 轴的关系。此外，用于生成 3D 图像报告的基于多模态大语言模型的方法面临着重大限制，特别是在图像编码器表示 3D 结构和生成内容中出现的幻觉的能力方面。为了应对这些挑战，我们提出了 3D-CT-GPT++ 模型。该模型集成了优化的 3D 图像编码器 CTViT-V，专为胸部 CT 扫描而设计，并建立在 LLaVA-1.5 架构之上。此外，我们引入了 \textit{直接偏好优化（DPO）}，其中 GPT-4 用于对完全微调（SFT）模型的输出进行评分，为后续 DPO 训练创建偏好数据集。 DPO 显着减少了报告生成过程中的幻觉，确保生成的报告更符合临床需求。我们在高质量的私人和公共数据集上对模型进行了微调，以确保临床相关性。使用标准自然语言生成 (NLG) 评估指标（包括 BLEU、METEOR 和 ROUGE-L）进行了大量实验，以评估报告生成性能。实验结果表明，3D-CT-GPT++在准确性、流畅性和临床相关性方面显着优于现有方法，推进了3D医疗报告生成的自动化。


## 45. Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision
在弱监督下，迭代标签细化比偏好优化更重要  
### 关键字
* Unreliable Human Supervision
* LM Post-training
* Scalable Oversight
### 主要内容
#### 对预训练LLM的后期训练提出问题：
PPO/RLHF/DPO在SFT之后进行优化，随着 LM 的能力变得更强，他们所承担的任务变得更难监督。在不可靠的监督下，培训后是否仍然有效？
#### 实验现象
在不可靠监督的情况下，SFT 仍然保留了一定的有效性，但 DPO 未能将模型改进到超越 SFT。
#### 给解决方案
1. 迭代标签细化（ILR）作为监督不可靠的 RLHF 的替代品
2. ILR 通过使用比较反馈直接改进 SFT 数据，以决定是否应该用模型生成的替代方案取代人类演示，然后通过 SFT 根据更新的数据重新训练模型。 
### 相关链接
* <a href="./papers/9316_Iterative_Label_Refinemen.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=q5EZ7gKcnW">ICLR链接</a>

### 摘要
Language model (LM) post-training relies on two stages of human supervision: task demonstrations for supervised finetuning (SFT), followed by preference comparisons for reinforcement learning from human feedback (RLHF) via algorithms like proximal preference optimization (PPO) or direct preference optimization (DPO). As LMs become more capable, the tasks they are given become harder to supervise. Will post-training remain effective under unreliable supervision? To test this, we simulate unreliable demonstrations and comparison feedback using small LMs and time-constrained humans. We find that in the presence of unreliable supervision, SFT still retains some effectiveness, but DPO fails to improve the model beyond SFT. To address this, we propose iterative label refinement (ILR) as a replacement for RLHF with unreliable supervision. ILR directly improves the SFT data by using comparison feedback to decide whether human demonstrations should be replaced by model-generated alternatives, then retrains the model via SFT on the updated data. SFT+ILR outperforms SFT+DPO on several tasks with LM-simulated unreliable supervision (math, coding, safe instruction-following), with results further verified by human experiments on instruction-following. Our findings suggest that as LMs take on complex tasks where human supervision is unreliable, RLHF may no longer be the best use of human comparison feedback; instead, it is better to direct feedback towards improving the training data rather than continually training the model.
语言模型 (LM) 后期训练依赖于人类监督的两个阶段：监督微调 (SFT) 的任务演示，然后通过近端偏好优化 (PPO) 或直接偏好等算法根据人类反馈 (RLHF) 进行强化学习的偏好比较优化（DPO）。随着 LM 的能力变得更强，他们所承担的任务变得更难监督。在不可靠的监督下，培训后是否仍然有效？为了测试这一点，我们使用小型 LM 和时间有限的人类来模拟不可靠的演示和比较反馈。我们发现，在不可靠监督的情况下，SFT 仍然保留了一定的有效性，但 DPO 未能将模型改进到超越 SFT。为了解决这个问题，我们提出迭代标签细化（ILR）作为监督不可靠的 RLHF 的替代品。 ILR 通过使用比较反馈直接改进 SFT 数据，以决定是否应该用模型生成的替代方案取代人类演示，然后通过 SFT 根据更新的数据重新训练模型。 SFT+ILR 在 LM 模拟的不可靠监督（数学、编码、安全指令跟踪）的多项任务上优于 SFT+DPO，并且结果通过指令跟踪的人体实验进一步验证。我们的研究结果表明，随着 LM 承担人类监督不可靠的复杂任务，RLHF 可能不再是人类比较反馈的最佳用途；相反，最好将反馈直接用于改进训练数据，而不是持续训练模型。


## 46. Scalable Preference Learning for Large Language Models via Convex Optimization
通过凸优化实现大型语言模型的可扩展偏好学习  
### 关键字
* LLM
* Preference Learning
* Convex  Optimization

### 主要内容
#### 更轻量的DPO
* 实现这一目标的关键是利用神经网络的凸优化重构，并减少对复制参考模型的依赖。
* 目标是更快地收敛到更好的最优性解决方案，以及生成语言任务的底层优化景观的更高可解释性。
* 使用交替方向乘法器 (ADMM) 来解决此优化问题，以提高并行化效率，并在 JAX 中实现此方法以解除实验中的内存限制。

### 相关链接
* <a href="./papers/13320_Accelerated_Preference_O.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=EVZnnhtMNX">ICLR链接</a>

### 摘要
Fine-tuning large language models (LLMs) for alignment with human preferences have become a key factor in the success of models like ChatGPT and Gemini, which are now integral to mainstream use. Many effective techniques are based on Reinforcement Learning from Human Feedback (RLHF), yet are complex, unstable, and expensive to implement. Recently, Direct Preference Optimization (DPO) offers an accessible alternative by simplifying the objective and training a policy model using a frozen, copied reference model to provide a stable training benchmark. In this paper, we develop an even more lightweight DPO based algorithm that operates on a single GPU. The key to achieving this is leveraging the convex optimization reformulation of neural networks, and reducing the dependence on copying the reference model. Our aim is to provide faster convergence to solutions of better optimality, and higher interpretability of the underlying optimization landscape for generative language tasks. We use the Alternating Direction Method of Multipliers (ADMM) to solve this optimization problem in order to increase parallelization efficiency, and implement our methods in JAX to lift the memory constraints across experiments. We experiment on three datasets, including one synthetically generated educational dataset, to demonstrate the efficacy of our novel algorithm in a real world setting. Our method is comparable in user preference generation to DPO when tested on 17 human volunteers, despite being trained on one single RTX-4090 GPU using a smaller dataset.
微调大型语言模型 ( LLMs ) 以符合人类偏好已成为 ChatGPT 和 Gemini 等模型成功的关键因素，这些模型现已成为主流使用的组成部分。许多有效的技术都是基于人类反馈强化学习 (RLHF)，但实施起来复杂、不稳定且昂贵。最近，直接偏好优化（DPO）通过简化目标并使用冻结的复制参考模型来训练策略模型来提供稳定的训练基准，从而提供了一种可行的替代方案。在本文中，我们开发了一种更轻量级的基于 DPO 的算法，该算法在单个 GPU 上运行。实现这一目标的关键是利用神经网络的凸优化重构，并减少对复制参考模型的依赖。我们的目标是更快地收敛到更好的最优性解决方案，以及生成语言任务的底层优化景观的更高可解释性。我们使用交替方向乘法器 (ADMM) 来解决此优化问题，以提高并行化效率，并在 JAX 中实现我们的方法以解除实验中的内存限制。我们对三个数据集（包括一个综合生成的教育数据集）进行实验，以证明我们的新颖算法在现实世界环境中的有效性。尽管是在单个 RTX-4090 GPU 上使用较小的数据集进行训练，但在 17 名人类志愿者身上进行测试时，我们的方法在用户偏好生成方面与 DPO 相当。

## 47. Hierarchical Preference Optimization: Learning to achieve goals via feasible subgoals prediction
分层偏好优化：学习通过可行的子目标预测来实现目标  
### 关键字
* HPO(Hierarchical PO)
* Preference Learning

### 主要内容
#### 针对问题：
复杂机器人控制任务时的非平稳性和不可行的子目标生成问题
#### 提出HPO
* 利用最大熵RL和token级DPO，消除对预先训练的参考策略的需求
* 在数学上，将HRL表述为双层优化问题，并将其转化为原始正则化 DPO 表述，确保可行的子目标生成并避免退化解决方案。
### 相关链接
* <a href="./papers/8804_Hierarchical_Preference_O.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=BsQTw0uPDX">ICLR链接</a>

### 摘要
This work introduces Hierarchical Preference Optimization (HPO), a novel approach to hierarchical reinforcement learning (HRL) that addresses non-stationarity and infeasible subgoal generation issues when solving complex robotic control tasks. HPO leverages maximum entropy reinforcement learning combined with token-level Direct Preference Optimization (DPO), eliminating the need for pre-trained reference policies that are typically unavailable in challenging robotic scenarios. Mathematically, we formulate HRL as a bi-level optimization problem and transform it into a primitive-regularized DPO formulation, ensuring feasible subgoal generation and avoiding degenerate solutions. Extensive experiments on challenging robotic navigation and manipulation tasks demonstrate HPO’s impressive performance, where HPO shows an improvement of up to 35% over the baselines. Furthermore, ablation studies validate our design choices, and quantitative analyses confirm HPO’s ability to mitigate non-stationarity and infeasible subgoal generation issues in HRL.
这项工作介绍了分层偏好优化（HPO），这是一种分层强化学习（HRL）的新方法，可解决解决复杂机器人控制任务时的非平稳性和不可行的子目标生成问题。 HPO 利用最大熵强化学习与token级直接偏好优化 (DPO) 相结合，消除了对预先训练的参考策略的需求，而这些策略在具有挑战性的机器人场景中通常是不可用的。在数学上，我们将 HRL 表述为双层优化问题，并将其转化为原始正则化 DPO 表述，确保可行的子目标生成并避免退化解决方案。针对具有挑战性的机器人导航和操作任务的大量实验证明了 HPO 的令人印象深刻的性能，其中 HPO 比基线提高了高达 35%。此外，消融研究验证了我们的设计选择，定量分析证实了 HPO 能够缓解 HRL 中的非平稳性和不可行的子目标生成问题。


## 48. Sample Efficient Alignment for LLMs
LLMs的有效调整示例  
### 关键字
* RLHF
* Online DAP
* LLM Alignment
* Sample Efficiency
### 主要内容
#### 研究主题
在给定预算的在线反馈的情况下，有效地将大型语言模型与人类偏好进行样本匹配的方法。 
#### 细节
1. 首先在上下文决斗强盗的框架中制定LLM对齐问题。 这种老虎机公式包含了最近出现的在线 RLHF / 在线 DPO 范式，自然地寻求样本高效的算法。
2. 研究了两种基于汤普森采样的主动探索算法，并阐明了它们的用例。

### 相关链接
* <a href="./papers/10057_Sample_Efficient_Alignme.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=Pf8i7cv2CH">ICLR链接</a>

### 摘要
We study methods for sample-efficiently aligning large language models with human preferences given budgeted online feedback. We first formulate the LLM alignment problem in the frame of contextual dueling bandits. This bandit formulation, subsuming the recently emerging online RLHF / online DPO paradigms, naturally quests for sample-efficient algorithms. Leveraging insights from bandits, we investigate two algorithms for active exploration based on Thompson sampling and shed light on their use cases. Our agent, termed as Sea (ample fficient lignment), is empirically validated with extensive experiments, across 3 scales (1B, 2.8B, 6.9B) and 3 preference learning algorithms (DPO, IPO, SimPO). The results show that Sea aligns the LLM with oracle's preferences highly sample-efficiently, surpassing recent SoTA methods. We will open-source our codebase to accelerate the research in this field.
我们研究了在给定预算的在线反馈的情况下，有效地将大型语言模型与人类偏好进行样本匹配的方法。 我们首先在上下文决斗强盗的框架中制定LLM对齐问题。 这种老虎机公式包含了最近出现的在线 RLHF / 在线 DPO 范式，自然地寻求样本高效的算法。 利用强盗的见解，我们研究了两种基于汤普森采样的主动探索算法，并阐明了它们的用例。我们的代理，称为Sea(sampling Efficient Alignment)，通过广泛的实验进行了实证验证，涵盖 3 个尺度（1B、2.8B、6.9B）和 3 种偏好学习算法（DPO、IPO、SimPO）。结果表明，Sea 使LLM与预言机的偏好高度一致，样本效率很高，超越了最近的 SoTA 方法。我们将开源我们的代码库以加速该领域的研究。


## 49. Direct Distributional Optimization for Provable Alignment of Diffusion Models
可证明扩散模型对齐的直接分布优化  

### 关键字
* Diffusion Model
* Optimization

### 主要内容
#### 扩散模型对齐新角度
从分布优化的角度引入了一种新颖的扩散模型对齐方法，同时提供严格的收敛保证。
#### 细节
1. 首先将问题表述为概率分布上的通用正则化损失最小化，并使用对偶平均方法直接优化分布
2. 通过 Doob 的得分函数近似来从学习的分布中进行采样h-变换技术。
    * Doob 变换是一种概率论技术，用于转化随机过程或分布，使其更适合于特定条件下的采样需求。
    * 所提出的框架得到严格的收敛保证和采样误差的端到端界限的支持，这意味着当准确地知道原始分布的分数时，从移位分布中采样的复杂性与等周条件无关。
    * 该框架广泛适用于一般分布优化问题，
### 相关链接
* <a href="./papers/4323_Direct_Distributional_Opt.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=Nvw2szDdmI">ICLR链接</a>

### 摘要
We introduce a novel alignment method for diffusion models from distribution optimization perspectives while providing rigorous convergence guarantees. We first formulate the problem as a generic regularized loss minimization over probability distributions and directly optimize the distribution using the Dual Averaging method. Next, we enable sampling from the learned distribution by approximating its score function via Doob's -transform technique. The proposed framework is supported by rigorous convergence guarantees and an end-to-end bound on the sampling error, which imply that when the original distribution's score is known accurately, the complexity of sampling from shifted distributions is independent of isoperimetric conditions. This framework is broadly applicable to general distribution optimization problems, including alignment tasks in Reinforcement Learning with Human Feedback (RLHF), Direct Preference Optimization (DPO), and Kahneman-Tversky Optimization (KTO). We empirically validate its performance on synthetic and image datasets using the DPO objective.
我们从分布优化的角度引入了一种新颖的扩散模型对齐方法，同时提供严格的收敛保证。 我们首先将问题表述为概率分布上的通用正则化损失最小化，并使用对偶平均方法直接优化分布。 接下来，我们通过 Doob 的得分函数近似来从学习的分布中进行采样 - 变换技术。所提出的框架得到严格的收敛保证和采样误差的端到端界限的支持，这意味着当准确地知道原始分布的分数时，从移位分布中采样的复杂性与等周条件无关。该框架广泛适用于一般分布优化问题，包括人类反馈强化学习 (RLHF)、直接偏好优化 (DPO) 和卡尼曼-特沃斯基优化 (KTO) 中的对齐任务。我们使用 DPO 目标凭经验验证其在合成数据集和图像数据集上的性能。


## 50. Learning Loss Landscapes in Preference Optimization
偏好优化中的学习损失情况  

### 关键字
* Preference Optimization
* Mirror Descent
    * (镜像下降通过将梯度下降过程“映射”到不同的空间，以适应问题的几何特性。与标准梯度下降不同，镜像下降会选择一种“镜像函数”或“距离函数”（如KL散度或Bregman散度）来重新定义更新步骤。)

### 主要内容
#### 主题
* 调查偏好数据集的特定属性（例如混合质量或噪声数据）如何影响偏好优化（PO）算法的性能。
* 在 MuJoCo (Multi-Joint dynamics with Contact) 环境中进行的实验揭示了最先进的 PO 方法性能显着下降的几种情况
#### 基于镜像下降的PO
* 针对镜像映射的特定选择恢复直接偏好优化（DPO）和优势比偏好优化（ORPO）等现有方法
* 采用进化策略来发现能够处理已识别问题场景的新损失函数。这些新的损失函数在多个任务中显着提高了 DPO 和 ORPO 的性能。
* 通过应用发现的损失函数使用混合质量数据微调大型语言模型来展示此方法的泛化能力，它们的性能优于 ORPO。
### 相关链接
* <a href="./papers/2402_Learning_Loss_Landscapes_.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=TU5ApbbeDZ">ICLR链接</a>

### 摘要
We present an empirical study investigating how specific properties of preference datasets, such as mixed-quality or noisy data, affect the performance of Preference Optimization (PO) algorithms. Our experiments, conducted in MuJoCo environments, reveal several scenarios where state-of-the-art PO methods experience significant drops in performance. To address this issue, we introduce a novel PO framework based on mirror descent, which can recover existing methods like Direct Preference Optimization (DPO) and Odds-Ratio Preference Optimization (ORPO) for specific choices of the mirror map. Within this framework, we employ evolutionary strategies to discover new loss functions capable of handling the identified problematic scenarios. These new loss functions lead to significant performance improvements over DPO and ORPO across several tasks. Additionally, we demonstrate the generalization capability of our approach by applying the discovered loss functions to fine-tuning large language models using mixed-quality data, where they outperform ORPO.
我们提出了一项实证研究，调查偏好数据集的特定属性（例如混合质量或噪声数据）如何影响偏好优化（PO）算法的性能。我们在 MuJoCo 环境中进行的实验揭示了最先进的 PO 方法性能显着下降的几种情况。为了解决这个问题，我们引入了一种基于镜像下降的新型 PO 框架，它可以针对镜像映射的特定选择恢复直接偏好优化（DPO）和优势比偏好优化（ORPO）等现有方法。在此框架内，我们采用进化策略来发现能够处理已识别问题场景的新损失函数。这些新的损失函数在多个任务中显着提高了 DPO 和 ORPO 的性能。此外，我们通过应用发现的损失函数使用混合质量数据微调大型语言模型来展示我们方法的泛化能力，它们的性能优于 ORPO。

## 51. Diffusion Preference Alignment via Relative Text-Image Contrast
通过相对文本图像对比度进行扩散偏好对齐   

### 关键字
* Diffusion Model
* Human Preference Alignment
* Fine-Tuning

### 主要内容
#### 现有的Diffusion-DPO
最初在单个文本提示的扩散模型中尝试了成对偏好学习。
#### 提出Diff-Contrast
* 旨在使基于扩散的 T2I 模型与人类偏好保持一致
* 利用具有相同提示的提示图像对以及跨不同模态语义相关的提示图像对
#### 提出风格对齐style-alignment
以解决当前人类偏好对齐评估相关的高成本、低可重复性和可解释性差的问题。

### 相关链接
* <a href="./papers/4840_Diffusion_Preference_Alig.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=rH6IZIXqZG">ICLR链接</a>

### 摘要
Aligning Large Language Models (LLMs) to human preferences has become a prominent area of research within language modeling. However, the application of preference learning to image generation in Text-to-Image (T2I) models remains relatively unexplored. One approach, Diffusion-DPO, initially experimented with pairwise preference learning in diffusion models for individual text prompts. We propose Diff-contrast, a novel method designed to align diffusion-based T2I models with human preferences. This method utilizes both prompt-image pairs with identical prompts and those that are semantically related across different modalities. Additionally, we introduced a new evaluation task, style alignment, to address the issues of high cost, low reproducibility, and poor interpretability associated with current evaluations of human preference alignment. Our results show that Diff-contrast surpasses existing techniques, e.g. Diffusion-DPO, in tuning Stable Diffusion versions 1.5 and XL-1.0 across both automated evaluations of human preference and style alignment.
使大型语言模型 ( LLMs ) 与人类偏好保持一致已成为语言建模中的一个突出研究领域。然而，偏好学习在文本到图像（T2I）模型中图像生成的应用仍然相对未经探索。一种方法，Diffusion-DPO，最初在单个文本提示的扩散模型中尝试了成对偏好学习。我们提出了 Diff-contrast，这是一种新颖的方法，旨在使基于扩散的 T2I 模型与人类偏好保持一致。该方法利用具有相同提示的提示图像对以及跨不同模态语义相关的提示图像对。此外，我们引入了一种新的评估任务——风格对齐，以解决当前人类偏好对齐评估相关的高成本、低可重复性和可解释性差的问题。我们的结果表明，Diff-contrast 超越了现有技术，例如 Diffusion-DPO，在人类偏好和风格对齐的自动评估方面调整稳定扩散版本 1.5 和 XL-1.0。


## 52. Understanding Alignment in Multimodal LLMs: A Comprehensive Study
了解多模式LLMs的一致性：一项综合研究  
### 关键字
* Foundation Models
* MLLMs
* Alignment
* Image Understanding
### 主要内容
#### 针对MLLM的偏好对齐
* 应对幻觉挑战：
    * 幻觉可以通过陈述不正确的事实而发生
    * 还可以通过产生与图像内容不一致的响应而发生
* MLLM 对齐的主要目标是鼓励这些模型将响应与图像信息更紧密地对齐
* 希望寻找不同偏好对齐方法(DPO, PPO等)中贡献大的元素
#### 细节
分析 MLLM 中偏好调整的各个方面
1. 将对齐算法分为两类：
    * 离线（例如 DPO）
    * 在线（例如 online-DPO）
2. 表明结合离线和在线方法可以在某些场景下提高模型的性能
3. 回顾各种已发布的多模式偏好数据集，并讨论其构建细节如何影响模型性能
4. 基于这些见解，引入了一种创建多模态偏好数据的新方法，称为`偏差驱动幻觉采样`（BDHS），它既不需要额外的注释，也不需要外部模型，并表明它可以实现与之前发布的多模态模型对齐工作竞争的性能
### 相关链接
* <a href="./papers/3243_Understanding_Alignment_i.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=49qqV4NTdy">ICLR链接</a>

### 摘要
Preference alignment has become a crucial component in enhancing the performance of Large Language Models (LLMs), yet its impact in Multimodal Large Language Models (MLLMs) remains comparatively underexplored. Similar to language models, MLLMs for image understanding tasks encounter challenges like hallucination. In MLLMs, hallucination can occur not only by stating incorrect facts but also by producing responses that are inconsistent with the image content. A primary objective of alignment for MLLMs is to encourage these models to align responses more closely with image information. Recently, multiple works have introduced preference datasets for MLLMs and examined different alignment methods, including Direct Preference Optimization (DPO) and Proximal Policy Optimization (PPO). However, due to variations in datasets, base model types, and alignment methods, it remains unclear which specific elements contribute most significantly to the reported improvements in these works. In this paper, we independently analyze each aspect of preference alignment in MLLMs. We start by categorizing the alignment algorithms into two groups, offline (such as DPO), and online (such as online-DPO), and show that combining offline and online methods can improve the performance of the model in certain scenarios. We review a variety of published multimodal preference datasets and discuss how the details of their construction impact model performance. Based on these insights, we introduce a novel way of creating multimodal preference data called Bias-Driven Hallucination Sampling (BDHS) that needs neither additional annotation nor external models, and show that it can achieve competitive performance to previously published alignment work for multimodal models across a range of benchmarks.
偏好对齐已成为提高大型语言模型 ( LLMs ) 性能的关键组成部分，但其对多模态大型语言模型 (MLLM) 的影响仍相对未得到充分研究。与语言模型类似，用于图像理解任务的 MLLM 也会遇到幻觉等挑战。在 MLLM 中，幻觉不仅可以通过陈述不正确的事实而发生，还可以通过产生与图像内容不一致的响应而发生。 MLLM 对齐的主要目标是鼓励这些模型将响应与图像信息更紧密地对齐。最近，多项工作引入了 MLLM 的偏好数据集，并研究了不同的对齐方法，包括直接偏好优化（DPO）和近端策略优化（PPO）。然而，由于数据集、基础模型类型和对齐方法的变化，目前尚不清楚哪些特定元素对这些工作中报告的改进贡献最大。在本文中，我们独立分析了 MLLM 中偏好调整的各个方面。我们首先将对齐算法分为两类：离线（例如 DPO）和在线（例如 online-DPO），并表明结合离线和在线方法可以在某些场景下提高模型的性能。我们回顾了各种已发布的多模式偏好数据集，并讨论其构建细节如何影响模型性能。基于这些见解，我们引入了一种创建多模态偏好数据的新方法，称为偏差驱动幻觉采样（BDHS），它既不需要额外的注释，也不需要外部模型，并表明它可以实现与之前发布的多模态模型对齐工作竞争的性能。一系列基准。

## 53. A Novel Soft Alignment Approach for Language Models with Explicit Listwise Rewards
具有显式列表奖励的语言模型的新颖软对齐方法  

### 关键字
* LLM
* Preference Optimization
* Listwise Optimization Objective

### 主要内容
#### DPO的特点
主要是针对成对偏好数据量身定制的，其中的奖励是隐式定义的，而不是显示给出的
#### 发掘LLM对齐的通用框架
利用新颖的优化目标来弥合处理`奖励数据集`与`用标量偏好分数明确注释的响应列表`之间的差距
#### SPO(Soft Preference Optimization)
* 可以从奖励数据和偏好数据中直接提取 LM 策略
* SPO 的核心是一种新颖的列表偏好优化目标，具有指数对数函数形式和自适应损失系数，将列表偏好信号注入大语言模型中。
### 相关链接
* <a href="./papers/8934_A_Novel_Soft_Alignment_Ap.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=28TLorTMnP">ICLR链接</a>

### 摘要
Existing alignment methods, such as Direct Preference Optimization (DPO), are mainly tailored for pairwise preference data where rewards are implicitly defined rather than explicitly given. In this paper, we introduce a general framework for large language model alignment, leveraging a novel optimization objective to bridge the gap in handling reward datasets with a list of responses explicitly annotated with scalar preferences scores.
Our work comprise a novel algorithm, soft preference optimization, SPO, which enables the direct extraction of an LM policy from reward data as well as preference data. The core of SPO is a novel listwise preference optimization objective with the exponential-logarithm function form and a adaptive loss coefficient that inject listwise preference signals into the large language model.
We evaluate our methods in both reward and preference settings with Mistral models in different sizes. Experiments suggest that our method surpasses various preference baselines when reward datasets are available. We also find our method significantly outperforms DPO in complex reasoning tasks like math and coding.
现有的对齐方法，例如直接偏好优化（DPO），主要是针对成对偏好数据量身定制的，其中奖励是隐式定义的，而不是显式给出的。在本文中，我们介绍了大型语言模型对齐的通用框架，利用新颖的优化目标来弥合处理奖励数据集与用标量偏好分数明确注释的响应列表之间的差距。
我们的工作包括一种新颖的算法，软偏好优化，SPO，它可以从奖励数据和偏好数据中直接提取 LM 策略。 SPO 的核心是一种新颖的列表偏好优化目标，具有指数对数函数形式和自适应损失系数，将列表偏好信号注入大语言模型中。
我们使用不同大小的 Mistral 模型在奖励和偏好设置中评估我们的方法。实验表明，当奖励数据集可用时，我们的方法超越了各种偏好基线。我们还发现我们的方法在数学和编码等复杂推理任务中显着优于 DPO。


## 54. Annotation-Efficient Language Model Alignment via Diverse and Representative Response Texts
通过多样化和代表性的响应文本进行高效注释的语言模型对齐  

### 关键字
* LLM
* DPO

### 主要内容
#### 面对数据集的量与质的问题
在许多应用中获得大量偏好注释是很困难的。这就提出了如何使用有限的注释预算来创建有效的偏好数据集的问题。
#### AEPO(Annotation-Efficient PO)
* AEPO 没有详尽地注释对所有可用响应文本的偏好，而是选择一个响应子集，使可用响应的多样性和代表性最大化，然后注释对所选响应文本的偏好。
* 通过这种方式，AEPO 将注释预算集中在对较小的响应子集的标记偏好上。
* 相同注释预算下优于标准DPO
### 相关链接
* <a href="./papers/12975_Annotation_Efficient_Lan.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=JFk8F7w8Iz">ICLR链接</a>

### 摘要
Preference optimization is a standard approach to fine-tuning large language models to align with human preferences. The quantity, diversity, and representativeness of the preference dataset are critical to the effectiveness of preference optimization. However, obtaining a large amount of preference annotations is difficult in many applications. This raises the question of how to use the limited annotation budget to create an effective preference dataset. To this end, we propose Annotation-Efficient Preference Optimization (AEPO). Instead of exhaustively annotating preference over all available response texts, AEPO selects a subset of responses that maximizes diversity and representativeness from the available responses and then annotates preference over the selected ones. In this way, AEPO focuses the annotation budget on labeling preference over a smaller subset of responses. We evaluate the performance of Direct Preference Optimization (DPO) using AEPO and show that it outperforms models trained using a standard DPO with the same annotation budget.
偏好优化是微调大型语言模型以符合人类偏好的标准方法。偏好数据集的数量、多样性和代表性对于偏好优化的有效性至关重要。然而，在许多应用中获得大量偏好注释是很困难的。这就提出了如何使用有限的注释预算来创建有效的偏好数据集的问题。为此，我们提出注释高效偏好优化（AEPO）。 AEPO 没有详尽地注释对所有可用响应文本的偏好，而是选择一个响应子集，使可用响应的多样性和代表性最大化，然后注释对所选响应文本的偏好。通过这种方式，AEPO 将注释预算集中在对较小的响应子集的标记偏好上。我们使用 AEPO 评估直接偏好优化 (DPO) 的性能，并表明它优于使用具有相同注释预算的标准 DPO 训练的模型。


## 55. Self-Augmented Preference Optimization: Off-Policy Paradigms for Language Model Alignment
自我增强偏好优化：语言模型对齐的离策略范式  
### 关键字
* LLM
* Fine-Tuning
### 主要内容
#### DPO数据集局限性
由于依赖静态的、预先收集的配对偏好数据而受到限制，这限制了它们的适应性和实际适用性。
#### 引入SAPO(self-augmentation PO)
* 有效且可扩展的训练范例，不需要现有的配对数据。
* 基于自主生成负面响应的自我博弈概念，进一步涉及离策略学习管道来改进数据探索和利用。
    * 采用指数移动平均 (EMA) 模型和重播缓冲区来实现响应片段的动态更新，从而有效地将实时反馈与历史数据洞察相结合。

### 相关链接
* <a href="./papers/8397_Self_Augmented_Preference.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=yizEOJVFFd">ICLR链接</a>

### 摘要
Traditional language model alignment methods, such as Direct Preference Optimization (DPO), are limited by their dependence on static, pre-collected paired preference data, which restricts their adaptability and practical applicability. To address this limitation, we introduce Self-Augmented Preference Optimization (SAPO), an effective and scalable training paradigm without the need of existing paired data. Built upon the self-play concept that autonomously generate negative responses, we further involve the off-policy learning pipeline to improve the data exploration and exploitation. Specifically, we employ an Exponential Moving Average (EMA) model along with a replay buffer to enable dynamic updates of response segments, effectively integrating real-time feedback with historical data insights. Our comprehensive evaluations of the LLaMA3-8B and Mistral-7B models across benchmarks—including the Open LLM Leaderboard, IFEval, AlpacaEval 2.0, and MT-Bench—demonstrate that SAPO matches or surpasses established offline contrastive baselines, such as DPO and Odds Ratio Preference Optimization (ORPO), and outperforms offline self-play methods like SPIN.
传统的语言模型对齐方法，例如直接偏好优化（DPO），由于依赖静态的、预先收集的配对偏好数据而受到限制，这限制了它们的适应性和实际适用性。为了解决这个限制，我们引入了自我增强偏好优化（SAPO），这是一种有效且可扩展的训练范例，不需要现有的配对数据。基于自主生成负面响应的自我博弈概念，我们进一步涉及离策略学习管道来改进数据探索和利用。具体来说，我们采用指数移动平均 (EMA) 模型和重播缓冲区来实现响应片段的动态更新，从而有效地将实时反馈与历史数据洞察相结合。我们跨基准（包括 Open LLM Leaderboard、IFEval、AlpacaEval 2.0 和 MT-Bench）对 LLaMA3-8B 和 Mistral-7B 模型进行的综合评估表明，SAPO 匹配或超过既定的离线对比基线，例如 DPO 和优势比偏好优化（ORPO），并且优于 SPIN 等离线自我对战方法。


## 56. RRM: Robust Reward Model Training Mitigates Reward Hacking
RRM：强大的奖励模型训练可减少奖励黑客攻击  
### 关键字
* Reward Model
* RLHF
* Alignment
### 主要内容
#### 针对Reward Hacking
* Agent通过操纵其环境或利用漏洞来获得奖励，而不是通过实际完成预期的任务目标。
* 传统的 RM 训练依赖于与特定Prompts相关的响应对，很难将提示驱动的偏好与独立于提示的工件（例如响应长度和格式）区分开来。
* 揭示了当前 RM 训练方法的一个基本局限性，即 RM 在确定偏好时无法有效地区分上下文信号和不相关的工件。
#### 引入因果框架
该框架可以学习独立于这些工件的偏好，并提出一种旨在消除它们的新颖的数据增强技术。
### 相关链接
* <a href="./papers/2424_RRM_Robust_Reward_Model_T.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=88AS5MQnmC">ICLR链接</a>

### 摘要
Reward models (RMs) play a pivotal role in aligning large language models (LLMs) with human preferences. However, traditional RM training, which relies on response pairs tied to specific prompts, struggles to disentangle prompt-driven preferences from prompt-independent artifacts, such as response length and format. In this work, we expose a fundamental limitation of current RM training methods, where RMs fail to effectively distinguish between contextual signals and irrelevant artifacts when determining preferences. To address this, we introduce a causal framework that learns preferences independent of these artifacts and propose a novel data augmentation technique designed to eliminate them. Extensive experiments show that our approach successfully filters out undesirable artifacts, yielding a more robust reward model (RRM). Our RRM improves the performance of a pairwise reward model trained on Gemma-2-9b-it, on Reward-Bench, increasing accuracy from 80.61% to 84.15%. Additionally, we train two DPO policies using both the RM and RRM, demonstrating that the RRM significantly enhances DPO-aligned policies, improving MT-Bench scores from 7.27 to 8.31 and length-controlled win-rates in AlpacaEval-2 from 33.46% to 52.49%.
奖励模型 (RM) 在使大型语言模型 ( LLMs ) 与人类偏好保持一致方面发挥着关键作用。然而，传统的 RM 训练依赖于与特定提示相关的响应对，很难将提示驱动的偏好与独立于提示的工件（例如响应长度和格式）区分开来。在这项工作中，我们揭示了当前 RM 训练方法的一个基本局限性，即 RM 在确定偏好时无法有效地区分上下文信号和不相关的工件。为了解决这个问题，我们引入了一个因果框架，该框架可以学习独立于这些工件的偏好，并提出一种旨在消除它们的新颖的数据增强技术。大量的实验表明，我们的方法成功地过滤掉了不需要的工件，从而产生了更强大的奖励模型（RRM）。我们的 RRM 提高了 Reward-Bench 上在 Gemma-2-9b-it 上训练的成对奖励模型的性能，将准确度从 80.61% 提高到 84.15%。此外，我们使用 RM 和 RRM 训练两个 DPO 策略，证明 RRM 显着增强了 DPO 一致策略，将 MT-Bench 分数从 7.27 提高到 8.31，并将 AlpacaEval-2 中的长度控制胜率从 33.46% 提高到 52.49 %。


## 57. A Novel Listwise Alignment Approach for Language Models with Explicit Rewards
具有显式奖励的语言模型的新颖列表对齐方法  
### 关键字
* LLMs
* Preference Alignment
* Listwise Optimization Objective
    * 对排序模型进行训练的一种方法。它将所有训练样本当作一个整体列表来优化，而不是单个样本或样本对。
    * 具体来说，模型会直接学习如何优化整个列表的排序性能指标（如NDCG、MRR等）。这种方法与pairwise（对式）或pointwise（点式）优化不同，listwise目标关注整个候选列表的排序结构，因此在学习复杂排序关系时更为有效。
### 主要内容
#### DPO只能接受成对偏好数据
其中奖励是推断出来的，而不是明确提供的
#### 此工作
* 提出了一个综合框架，通过引入一个新的优化目标来调整LLMs，该目标有助于奖励数据集的处理，其中包括明确标有标量偏好分数的响应列表。
* 开发了一种称为软偏好优化 (LPO) 的新颖算法，该算法允许从奖励和偏好数据集中直接推导LLM策略。 
    * LPO 的核心是使用`指数对数函数`和`自适应损失系数`制定的独特列表偏好优化目标，它有效地将列表偏好信号集成到LLM中。
### 相关链接
* <a href="./papers/10079_A_Novel_Listwise_Alignme.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=Ek50sQQI1w">ICLR链接</a>

### 摘要
Existing alignment techniques, including Direct Preference Optimization (DPO), are primarily designed for pairwise preference data where rewards are inferred rather than explicitly provided. In this paper, we propose a comprehensive framework for aligning large language models (LLMs) by introducing a new optimization objective that facilitates the processing of reward datasets, which consist of a list of responses explicitly marked with scalar preference scores. Our contribution includes the development of a novel algorithm, termed Soft Preference Optimization (LPO), which allows for the direct derivation of an LLM policy from both reward and preference datasets. At the heart of LPO is a unique listwise preference optimization objective formulated using an exponential-logarithmic function and an adaptive loss coefficient, which effectively integrates listwise preference signals into the LLM. We assess the efficacy of our approach under both reward and preference scenarios using different sizes of Mistral models. Experimental results indicate that our method outperforms several preference-based benchmarks, particularly when reward datasets are utilized. Additionally, our method demonstrates a significant advantage over DPO in intricate reasoning tasks, such as mathematical problem-solving and coding.
现有的对齐技术，包括直接偏好优化（DPO），主要是为成对偏好数据设计的，其中奖励是推断出来的，而不是明确提供的。在本文中，我们提出了一个综合框架，通过引入一个新的优化目标来调整大型语言模型（ LLMs ），该目标有助于奖励数据集的处理，其中包括明确标有标量偏好分数的响应列表。我们的贡献包括开发了一种称为软偏好优化 (LPO) 的新颖算法，该算法允许从奖励和偏好数据集中直接推导LLM策略。 LPO 的核心是使用指数对数函数和自适应损失系数制定的独特列表偏好优化目标，它有效地将列表偏好信号集成到LLM中。我们使用不同规模的米斯特拉尔模型评估了我们的方法在奖励和偏好场景下的有效性。实验结果表明，我们的方法优于几种基于偏好的基准，特别是在使用奖励数据集时。此外，我们的方法在复杂的推理任务（例如数学问题解决和编码）中表现出优于 DPO 的显着优势。


## 58. No Preference Left Behind: Group Distributional Preference Optimization
不遗漏任何偏好：群体分配偏好优化  
### 关键字
* Preference Alignment
* LLMs
* Fairness
* Group Preferences
### 主要内容
#### DPO等在反映群体偏好时的局限性
* 一群人的偏好并不统一，而是遵循一定的分布。
* 虽然DPO等现有的对齐方法试图引导模型反映人类偏好，但它们很难捕获群体内的分布多元偏好。这些方法往往偏向于主导偏好，忽视了意见的多样性，尤其是当偏好出现冲突时。
#### GDPO(Group DPO)
* 通过结合塑造个人偏好的信念概念，使语言模型与群体内的偏好分布保持一致。 
* GDPO 使用群体信念分布的统计估计来校准语言模型，并将模型与信念条件偏好对齐，从而提供比传统方法更具包容性的对齐框架。
    * 在使用合成可控意见生成和现实世界电影评论数据集的实验中，表示DPO是有偏的而GDPO可以弥合这种差距
### 相关链接
* <a href="./papers/13392_No_Preference_Left_Behin.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=bgpNJBD6Va">ICLR链接</a>

### 摘要
Preferences within a group of people are not uniform but follow a distribution. While existing alignment methods like Direct Preference Optimization (DPO) attempt to steer models to reflect human preferences, they struggle to capture the distributional pluralistic preferences within a group. These methods often skew toward dominant preferences, overlooking the diversity of opinions, especially when conflicting preferences arise. To address this issue, we propose Group Distribution Preference Optimization (GDPO), a novel framework that aligns language models with the distribution of preferences within a group by incorporating the concept of beliefs that shape individual preferences. GDPO calibrates a language model using statistical estimation of the group's belief distribution and aligns the model with belief-conditioned preferences, offering a more inclusive alignment framework than traditional methods. In experiments using both synthetic controllable opinion generation and real-world movie review datasets, we show that DPO fails to align with the targeted belief distributions, while GDPO consistently reduces this alignment gap during training. Additionally, our evaluation metrics demonstrate that GDPO outperforms existing approaches in aligning with group distributional preferences, marking a significant advance in pluralistic alignment.
一群人的偏好并不统一，而是遵循一定的分布。虽然直接偏好优化（DPO）等现有的对齐方法试图引导模型反映人类偏好，但它们很难捕获群体内的分布多元偏好。这些方法往往偏向于主导偏好，忽视了意见的多样性，尤其是当偏好出现冲突时。为了解决这个问题，我们提出了群体分布偏好优化（GDPO），这是一种新颖的框架，通过结合塑造个人偏好的信念概念，使语言模型与群体内的偏好分布保持一致。 GDPO 使用群体信念分布的统计估计来校准语言模型，并将模型与信念条件偏好对齐，从而提供比传统方法更具包容性的对齐框架。在使用合成可控意见生成和现实世界电影评论数据集的实验中，我们表明 DPO 未能与目标信念分布保持一致，而 GDPO 在训练过程中不断减少这种对齐差距。此外，我们的评估指标表明，GDPO 在符合群体分配偏好方面优于现有方法，标志着多元化一致性的重大进步。


## 59. HyperDPO: Hypernetwork-based Multi-Objective Fine-Tuning Framework
HyperDPO：基于超网络的多目标微调框架  
### 关键字
* DPO
* Multi-Objective Optimization
* Hypernetwork
    * 生成网络权重的网络结构。超网络不直接处理任务数据，而是通过生成其他模型（称为主网络或基础模型）的参数来间接优化任务。
    * 这种结构允许超网络根据不同任务或目标动态生成适合的权重，从而在多目标优化（如多任务学习或对齐任务）中展现出灵活性和高效性。它能在不同目标之间进行权重调节，而无需直接修改主网络的架构。
* Alignment

### 主要内容
#### 针对多目标微调任务(MOFT)
使用同时标记不同目标的数据集对现有模型进行微调
#### HyperDPO
* 使用HyperNetwork超网络拓展DPO
* 该技术最初是为与偏好数据进行高效的LLM对齐而开发的，以适应 MOFT 设置。
* 通过用 Plackett-Luce 模型替换 DPO 中的 Bradley-Terry-Luce 模型，该框架能够处理涉及列表排序数据集的各种 MOFT 任务。
    * BTL模型主要用于处理二选一
    * PL模型则进行扩展，适用于多项排序的情形，不局限于成对比较
* HyperDPO 具有高效的一次性训练过程来分析辅助目标的 Pareto 前沿，并提供灵活的训练后权衡控制。
#### Hyper Prompt Tuning设计
提出一种新颖的Hyper Prompt Tuning设计，该设计可以将跨目标的连续权重传递给基于变压器的模型，而无需改变其架构。
### 相关链接
* <a href="./papers/5520_HyperDPO_Hypernetwork_bas.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=qBKA2844I4">ICLR链接</a>

### 摘要
In LLM alignment and many other ML applications, one often faces the Multi-Objective Fine-Tuning (MOFT) problem, i.e. fine-tuning an existing model with datasets labeled w.r.t. different objectives simultaneously. To address the challenge, we propose the HyperDPO framework, a hypernetwork-based approach that extends the Direct Preference Optimization (DPO) technique, originally developed for efficient LLM alignment with preference data, to accommodate the MOFT settings. By substituting the Bradley-Terry-Luce model in DPO with the Plackett-Luce model, our framework is capable of handling a wide range of MOFT tasks that involve listwise ranking datasets. Compared with previous approaches, HyperDPO enjoys an efficient one-shot training process for profiling the Pareto front of auxiliary objectives, and offers flexible post-training control over trade-offs. Additionally, we propose a novel Hyper Prompt Tuning design, that conveys continuous weight across objectives to transformer-based models without altering their architecture. We demonstrate the effectiveness and efficiency of the HyperDPO framework through its applications to various tasks, including Learning-to-Rank (LTR) and LLM alignment, highlighting its viability for large-scale ML deployments.
在LLM对齐和许多其他 ML 应用中，人们经常面临多目标微调 (MOFT)问题，即使用同时标记不同目标的数据集对现有模型进行微调。为了应对这一挑战，我们提出了HyperDPO框架，这是一种基于超网络的方法，它扩展了直接偏好优化 (DPO) 技术，该技术最初是为与偏好数据进行高效的LLM对齐而开发的，以适应 MOFT 设置。通过用 Plackett-Luce 模型替换 DPO 中的 Bradley-Terry-Luce 模型，我们的框架能够处理涉及列表排序数据集的各种 MOFT 任务。与以前的方法相比，HyperDPO 具有高效的一次性训练过程来分析辅助目标的 Pareto 前沿，并提供灵活的训练后权衡控制。此外，我们提出了一种新颖的Hyper Prompt Tuning设计，该设计可以将跨目标的连续权重传递给基于变压器的模型，而无需改变其架构。我们通过将 HyperDPO 框架应用于各种任务（包括学习排序 (LTR) 和LLM对齐）来展示其有效性和效率，强调其大规模 ML 部署的可行性。

## 60. Relative Preference Optimization: Enhancing LLM Alignment through Contrasting Responses across Identical and Diverse Prompts
相对偏好优化：通过对比相同和不同提示的响应来增强LLM一致性  
### 关键字
* LLM alignment
* Fine-Tuning
* Preferences
### 主要内容
#### DPO无法完全反映人类学习的复杂本质
人类学习通常涉及理解对相同问题和相似问题的不同反应。
#### RPO(Relative PO)
* RPO 旨在区分源自相同和相关提示的更受欢迎和更不受欢迎的响应。
* 它引入了对比加权机制，可以使用更广泛的偏好数据（包括配对和不配对的数据集）来调整LLMs 。
    * 扩展模型的学习能力，使其能够利用来自更多样化提示的见解。
### 相关链接
* <a href="./papers/8305_Relative_Preference_Optim.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=APDnmucgID">ICLR链接</a>

### 摘要
In the field of large language models (LLMs), aligning models with the diverse preferences of users is a critical challenge. Direct Preference Optimization (DPO) has played a key role in this area. It works by using pairs of preferences derived from the same prompts, and it functions without needing an additional reward model. However, DPO does not fully reflect the complex nature of human learning, which often involves understanding contrasting responses to not only identical but also similar questions. To overcome this shortfall, we propose Relative Preference Optimization (RPO). RPO is designed to discern between more and less preferred responses derived from both identical and related prompts. It introduces a contrastive weighting mechanism, enabling the tuning of LLMs using a broader range of preference data, including both paired and unpaired sets. This approach expands the learning capabilities of the model, allowing it to leverage insights from a more varied set of prompts. Experiments in both paired and unpaired dataset settings, including tasks like dialogue, summarization, and general evaluation benchmarks, demonstrate RPO's superior ability to align LLMs with user preferences and enhance adaptability during training.
在大型语言模型（ LLMs ）领域，使模型与用户的不同偏好保持一致是一项严峻的挑战。直接偏好优化（DPO）在这一领域发挥了关键作用。它的工作原理是使用源自相同提示的偏好对，并且无需额外的奖励模型即可发挥作用。然而，DPO 并没有完全反映人类学习的复杂本质，人类学习通常涉及理解对相同问题和相似问题的不同反应。为了克服这一不足，我们提出了相对偏好优化（RPO）。 RPO 旨在区分源自相同和相关提示的更受欢迎和更不受欢迎的响应。它引入了对比加权机制，可以使用更广泛的偏好数据（包括配对和不配对的数据集）来调整LLMs 。这种方法扩展了模型的学习能力，使其能够利用来自更多样化提示的见解。在配对和不配对数据集设置中进行的实验（包括对话、总结和一般评估基准等任务）证明了 RPO 能够使LLMs与用户偏好保持一致并增强训练期间的适应性。


## 61. PAFT: A Parallel Training Paradigm for Effective LLM Fine-Tuning
PAFT：有效LLM微调的并行培训范式  
### 关键字
* LLM
* Fine-Tuning
* Alignment
* SFT
### 主要内容
#### SFT在PO的顺序可能会导致alignment tax
#### PAFT(PArallel Finetuning)
* 用于有效LLM微调的新并行训练范例：在各自的数据集上使用相同的预训练模型独立执行SFT和PO，然后两个模型通过参数融合得到最终模型，供下游应用使用
* 这项工作揭示了重要的发现，即像 DPO 这样的偏好对齐自然会产生稀疏模型，而 SFT 会产生自然密集模型，需要对其进行稀疏化才能有效进行模型合并。

### 相关链接
* <a href="./papers/8986_PAFT_A_Parallel_Training_.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=KHTkRhq2aB">ICLR链接</a>

### 摘要
Large language models (LLMs) have shown remarkable abilities in diverse natural language processing (NLP) tasks. The LLMs generally undergo supervised fine-tuning (SFT) followed by preference alignment to be usable in downstream applications. However, this sequential training pipeline leads to alignment tax that degrades the LLM performance.
This paper introduces PAFT, a new PArallel training paradigm for effective LLM Fine-Tuning, which independently performs SFT and preference alignment (e.g., DPO and ORPO, etc.) with the same pre-trained model on respective datasets. The model produced by SFT and the model from preference alignment are then merged into a final model by parameter fusing for use in downstream applications. This work reveals important findings that preference alignment like DPO naturally results in a sparse model while SFT leads to a natural dense model which needs to be sparsified for effective model merging. This paper introduces an effective interference resolution which reduces the redundancy by sparsifying the delta parameters. The LLM resulted from the new training paradigm achieved Rank #1 on the HuggingFace Open LLM Leaderboard. Comprehensive evaluation shows the effectiveness of the parallel training paradigm.
大型语言模型（ LLMs ）在各种自然语言处理（NLP）任务中表现出了卓越的能力。 LLMs通常会经过监督微调（SFT），然后进行偏好调整，以便在下游应用中使用。然而，这种连续的训练流程会导致对齐税，从而降低LLM表现。
本文介绍了 PAFT，一种用于有效LLM微调的新并行训练范例，它在各自的数据集上使用相同的预训练模型独立执行 SFT 和偏好对齐（例如 DPO 和 ORPO 等）。然后，SFT 生成的模型和偏好对齐的模型通过参数融合合并为最终模型，以供下游应用使用。这项工作揭示了重要的发现，即像 DPO 这样的偏好对齐自然会产生稀疏模型，而 SFT 会产生自然密集模型，需要对其进行稀疏化才能有效进行模型合并。本文介绍了一种有效的干扰解决方法，通过稀疏增量参数来减少冗余。 LLM源于新的培训模式，在 HuggingFace 开放LLM排行榜上排名第一。综合评价显示了并行训练范式的有效性。

## 62. SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety
SafeDPO：一种直接优化偏好并增强安全性的简单方法  

### 关键字
* Safety Alignment
* LLM Fine-Tuning
* Preferences 
* LLMs
* AI Safety
### 主要内容
#### 应对安全问题
* 使用RLHF中添加安全约束的方法往往复杂且常常不稳定，因为它们包含RLHF中的复杂程序以及安全限制所需的附加程序
#### SafeDPO
* 该算法旨在在策略学习的单个阶段中隐式优化安全对齐目标。
* 由此产生的算法可以通过仅引入一个额外的超参数来实现，其目的是进一步增强安全性，并对 DPO 实现进行少量修改。
* 因此，SafeDPO成功地消除了拟合奖励和成本模型以及在微调期间从语言模型中采样的必要性，同时仍然增强了LLMs的安全性。

### 相关链接
* <a href="./papers/5041_SafeDPO_A_Simple_Approach.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=MoJSnVZ59d">ICLR链接</a>

### 摘要
As large language models (LLMs) continue to advance and find applications across a growing number of fields, ensuring the safety of LLMs has become increasingly critical. To address safety concerns, recent studies have proposed integrating safety constraints into reinforcement learning from human feedback (RLHF). However, these approaches tend to be complex and often unstable, as they encompass complicated procedures in RLHF along with additional procedures required by the safety constraints. Inspired by direct preference optimization (DPO), we introduce a new algorithm called \textit{SafeDPO}, which is designed to implicitly optimize the safety alignment objective within a single stage of policy learning. The resulting algorithm can be implemented by introducing only one additional hyperparameter, which aims to further enhance safety, along with minor modifications to the DPO implementation. Consequently, SafeDPO successfully eliminates the necessity of fitting a reward and a cost model, as well as sampling from the language model during fine-tuning, while still enhancing the safety of LLMs. Finally, we demonstrate that SafeDPO achieves competitive performance compared to the current state-of-the-art safety alignment algorithm, both in terms of aligning with human preferences and improving safety.
随着大型语言模型 ( LLMs ) 的不断发展并在越来越多的领域找到应用，确保LLMs的安全变得越来越重要。为了解决安全问题，最近的研究提出将安全约束纳入人类反馈的强化学习（RLHF）中。然而，这些方法往往很复杂并且常常不稳定，因为它们包含 RLHF 中的复杂程序以及安全限制所需的附加程序。受直接偏好优化（DPO）的启发，我们引入了一种名为 \textit{SafeDPO} 的新算法，该算法旨在在策略学习的单个阶段中隐式优化安全对齐目标。由此产生的算法可以通过仅引入一个额外的超参数来实现，其目的是进一步增强安全性，并对 DPO 实现进行少量修改。因此，SafeDPO 成功地消除了拟合奖励和成本模型以及在微调期间从语言模型中采样的必要性，同时仍然增强了LLMs的安全性。最后，我们证明，与当前最先进的安全对齐算法相比，SafeDPO 在符合人类偏好和提高安全性方面都实现了具有竞争力的性能。


## 63. Multi-Step Preference Optimization via Two-Player Markov Games
通过两人马尔可夫博弈进行多步偏好优化  
### 关键字
* Multi-step Preference Optimization
* Two-player Markov game
* RLHF
* Optimistic Online Gradient Descent
    * 乐观在线梯度下降结合了乐观和梯度下降策略，常用于解决非凸优化和对抗性环境中的问题。
    * OGD使用估计的乐观偏差来加速收敛，有助于在竞争对手策略不断变化的情况下动态调整。
### 主要内容
#### DPO的局限性
* DPO将模型描述为一个Bandit问题，这限制了它们在多轮对话常见的现实场景中的适用性
* DPO 依赖于 Bradley-Terry 模型假设，该假设无法充分捕捉人类偏好的非传递性。
#### MPO和OMPO
* 通过将对齐问题建模为1-player Markov Game来解决这些挑战，其中每个player都寻求在对话的所有步骤中最大化他们对另一个玩家的获胜率
* 此MPO是建立在natural actor-critic framework之上的
* 基于乐观在线梯度下降算法进一步开发了 OMPO
* 理论上，对两种算法的收敛性进行了严格分析 
### 相关链接
* <a href="./papers/11193_Multi_Step_Preference_Op.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=NTNdRElwbp">ICLR链接</a>

### 摘要
Reinforcement Learning from Human Feedback (RLHF) has been highly successful in aligning large language models with human preferences. While prevalent methods like DPO have demonstrated strong performance, they frame interactions with the language model as a bandit problem, which limits their applicability in real-world scenarios where multi-turn conversations are common. Additionally, DPO relies on the Bradley-Terry model assumption, which does not adequately capture the non-transitive nature of human preferences. In this paper, we address these challenges by modeling the alignment problem as a two-player constant-sum Markov game, where each player seeks to maximize their winning rate against the other across all steps of the conversation. Our approach Multi-step Preference Optimization (MPO) is built upon the natural actor-critic framework. We further develop OMPO based on the optimistic online gradient descent algorithm. Theoretically, we provide a rigorous analysis for both algorithms on convergence and show that OMPO requires policy updates to converge to an -approximate Nash equilibrium. We also validate the effectiveness of our method through experiments on the multi-turn conversations dataset in MT-bench-101.
RLHF在使大型语言模型与人类偏好保持一致方面取得了巨大成功。虽然像 DPO 这样的流行方法已经表现出了强大的性能，但它们将与语言模型的交互视为强盗问题，这限制了它们在多轮对话常见的现实场景中的适用性。此外，DPO 依赖于 Bradley-Terry 模型假设，该假设无法充分捕捉人类偏好的非传递性。在本文中，我们通过将对齐问题建模为两人常和马尔可夫游戏来解决这些挑战，其中每个玩家都寻求在对话的所有步骤中最大化他们对另一个玩家的获胜率。我们的方法多步偏好优化（MPO）是建立在自然的演员评论家框架之上的。我们基于乐观在线梯度下降算法进一步开发了 OMPO。理论上，我们对两种算法的收敛性进行了严格的分析，并表明 OMPO 需要 政策更新趋于一致 - 近似纳什均衡。我们还通过 MT-bench-101 中的多轮对话数据集的实验验证了我们方法的有效性。


## 64. TODO: Enhancing LLM Alignment with Ternary Preferences
TODO：加强LLM与三元偏好的结合  

### 关键字
* LLMs
* Preference Alignment
* Ternary Preference

### 主要内容
#### DPO依赖二元模型
二元模型很难捕捉到人类偏好的复杂性，尤其是在存在嘈杂或不一致的标签以及频繁联系的情况下。
#### Tie(平局)-rank Oriented Bradley-Terry model (TOBT)
* 明确的包含平局，从而实现更细致的偏好表示，
* 在此基础上提出了面向平局的直接偏好优化(TODO)，利用 TOBT 的三元排序系统来改善偏好对齐。

### 相关链接
* <a href="./papers/7075_TODO_Enhancing_LLM_Alignm.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=utkGLDSNOk">ICLR链接</a>

### 摘要
Aligning large language models (LLMs) with human intent is critical for enhancing their performance across a variety of tasks. Standard alignment techniques, such as Direct Preference Optimization (DPO), often rely on the binary Bradley-Terry (BT) model, which can struggle to capture the complexities of human preferences—particularly in the presence of noisy or inconsistent labels and frequent ties. To address these limitations, we introduce the Tie-rank Oriented Bradley-Terry model (TOBT), an extension of the BT model that explicitly incorporates ties, enabling more nuanced preference representation. Building on this, we propose Tie-rank Oriented Direct Preference Optimization (TODO), a novel alignment algorithm that leverages TOBT's ternary ranking system to improve preference alignment. In evaluations on Mistral-7B and Llama 3-8B models, TODO consistently outperforms DPO in modeling preferences across both in-distribution and out-of-distribution datasets. Additional assessments using MT Bench and benchmarks such as Piqa, ARC-c, and MMLU further demonstrate TODO's superior alignment performance. Notably, TODO also shows strong results in binary preference alignment, highlighting its versatility and potential for broader integration into LLM alignment. The code for TODO is made publicly available.
将大型语言模型 ( LLMs ) 与人类意图保持一致对于提高其在各种任务中的性能至关重要。标准对齐技术，例如直接偏好优化 (DPO)，通常依赖于二元 Bradley-Terry (BT) 模型，该模型很难捕捉人类偏好的复杂性，尤其是在存在嘈杂或不一致的标签以及频繁联系的情况下。为了解决这些限制，我们引入了面向领带等级的 Bradley-Terry 模型 (TOBT)，这是 BT 模型的扩展，它明确地包含领带，从而实现更细致的偏好表示。在此基础上，我们提出了面向平局的直接偏好优化（TODO），这是一种新颖的对齐算法，利用 TOBT 的三元排序系统来改善偏好对齐。在对 Mistral-7B 和 Llama 3-8B 模型的评估中，TODO 在分布内和分布外数据集的建模偏好方面始终优于 DPO。使用 MT Bench 和 Piqa、ARC-c 和 MMLU 等基准进行的附加评估进一步证明了 TODO 卓越的对齐性能。值得注意的是，TODO 在二元偏好对齐方面也显示出强劲的结果，突出了其多功能性和更广泛地融入LLM对齐的潜力。 TODO 的代码已公开。


## 65. Data-Centric Human Preference Optimization with Rationales
以数据为中心的人类偏好优化及其基本原理  
### 关键字
* DPO
* Preference Learning
* Alignment

### 主要内容
#### 许多研究从算法层面优化RLHF
#### 此工作从数据层面优化
* 建议用机器生成的理由来丰富现有的偏好数据集，以解释选择背后的原因。
* 开发一个框架，用基本原理信息来增强当前的偏好学习方法。
* Rationale-enriched Preference Learning优势
    * 提高注释效率
    * 加速向高性能模型的收敛
    * 减少冗长偏差和幻觉
    * 有足够通用性
### 相关链接
* <a href="./papers/8553_Data_Centric_Human_Prefer.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=2Cg4YrsCMA">ICLR链接</a>

### 摘要
Reinforcement learning from human feedback plays a crucial role in aligning language models towards human preferences, traditionally represented through comparisons between pairs or sets of responses within a given context. While many studies have enhanced algorithmic techniques to optimize learning from such data, this work shifts focus to improving preference learning through a data-centric approach. Specifically, we propose enriching existing preference datasets with machine-generated rationales that explain the reasons behind choices. We develop a simple and principled framework to augment current preference learning methods with rationale information. Our comprehensive analysis highlights how rationales enhance learning efficiency. Extensive experiments reveal that rationale-enriched preference learning offers multiple advantages: it improves annotation efficiency, accelerates convergence to higher-performing models, and reduces verbosity bias and hallucination. Furthermore, this framework is versatile enough to integrate with various preference optimization algorithms. Overall, our findings highlight the potential of re-imagining data design for preference learning, demonstrating that even freely available machine-generated rationales can significantly boost performance across multiple dimensions.
RLHF在使语言模型符合人类偏好方面发挥着至关重要的作用，传统上通过在给定上下文中对或组响应之间的比较来表示。虽然许多研究增强了算法技术来优化从此类数据中的学习，但这项工作将重点转移到通过以数据为中心的方法来改进偏好学习。具体来说，我们建议用机器生成的理由来丰富现有的偏好数据集，以解释选择背后的原因。我们开发了一个简单而有原则的框架，用基本原理信息来增强当前的偏好学习方法。我们的全面分析强调了原理如何提高学习效率。大量实验表明，富含理论依据的偏好学习具有多种优势：它提高了注释效率，加速了向高性能模型的收敛，并减少了冗长偏差和幻觉。此外，该框架具有足够的通用性，可以与各种偏好优化算法集成。总体而言，我们的研究结果强调了重新想象偏好学习的数据设计的潜力，表明即使免费提供的机器生成的基本原理也可以显着提高多个维度的性能。


## 66. Declarative characterizations of direct preference alignment algorithms
直接偏好对齐算法的声明性特征  

### 关键字
* Neuro-Symbolic Modeling
* Logic
* Preference Learning
* RLHF
### 主要内容
#### 针对DPO变体的开发
* 原始 DPO 损失的许多新变体的开发，由于缺乏推理的技术和概念框架，理解这些最近提案之间的差异以及开发新的 DPA 损失函数仍然是一个艰巨的挑战。
#### 形式化DPA损失
尝试通过离散推理问题形式化 DPA 损失来解决这个问题。
1. 提出问题：给定现有的 DPA 损失，能否系统地导出表征其语义的符号表达式？
2. 展示了偏好学习的这种正式观点如何为 DPA 损失景观的规模和结构提供新的视角，从而不仅可以严格描述最近提案之间的关系，而且可以从第一原理中推导出新的损失函数。
3. 还将其正式发现与迄今为止未经测试的单一模型偏好损失函数类别的实证结果结合起来。
（我们的实验揭示了符号约束复杂性与相应损失的经验成功和训练动态之间的有趣联系，我们相信这些见解可以为致力于人工智能对齐的人工智能从业者提供有用的指导。
### 相关链接
* <a href="./papers/13235_Declarative_characteriza.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=7UKHNQIErp">ICLR链接</a>

### 摘要
Recent direct preference alignment algorithms (DPA), such as DPO, have shown great promise in aligning large language models to human preferences. While this has motivated the development on many new variants of the original DPO loss, understanding the the differences between these recent proposals, as well as developing new DPA loss functions, remains a formidable challenge given the lack of a technical and conceptual framework for reasoning about the underlying semantics of these algorithms. In this paper, we attempt to remedy this by formalizing DPA losses in terms of discrete reasoning problems. Specifically, we ask: Given an existing DPA loss, can we systematically derive a symbolic expression that characterizes its semantics? We show how this formal view of preference learning sheds new light on both the size and structure of DPA loss landscape, making it possible to not only rigorously characterize the relationships between recent proposals but to derive new loss functions from first principles. We also couple our formal findings with empirical results on a hitherto untested class of single model preference loss functions. Our experiments reveal interesting connections between symbolic constraint complexity and the empirical success and training dynamics of the corresponding losses, insights we believe can give useful guidance to AI practitioners working on AI alignment.
最近的直接偏好对齐算法（DPA），例如 DPO，在将大型语言模型与人类偏好对齐方面表现出了巨大的希望。虽然这推动了原始 DPO 损失的许多新变体的开发，但由于缺乏推理的技术和概念框架，理解这些最近提案之间的差异以及开发新的 DPA 损失函数仍然是一个艰巨的挑战。这些算法的底层语义。在本文中，我们尝试通过离散推理问题形式化 DPA 损失来解决这个问题。具体来说，我们要问：给定现有的 DPA 损失，我们能否系统地导出表征其语义的符号表达式？我们展示了偏好学习的这种正式观点如何为 DPA 损失景观的规模和结构提供新的视角，从而不仅可以严格描述最近提案之间的关系，而且可以从第一原理中推导出新的损失函数。我们还将我们的正式发现与迄今为止未经测试的单一模型偏好损失函数类别的实证结果结合起来。我们的实验揭示了符号约束复杂性与相应损失的经验成功和训练动态之间的有趣联系，我们相信这些见解可以为致力于人工智能对齐的人工智能从业者提供有用的指导。

## 67. Learning to Clarify: Multi-turn Conversations with Action-Based Contrastive Self-Training
学会澄清：通过基于行动的对比自我训练进行多轮对话  

### 关键字
* dialogue system
* conversation modeling
* reinforcement learning
* mixed-initiative interaction (互过程中，双方（人或系统）都可主动发起对话，提升互动的灵活性)
* LLM
* RLHF
* domain adaptation (模型从一个数据域（如特定场景）转移到新领域，并保持良好性能的过程)
* data-efficient learning
* clarification questions (在对话或任务中，为了解更详细信息而提问的问题，帮助提高沟通精确度)

### 主要内容
#### LLM agent的局限
* LLM agent可能仍然缺乏诸如消歧之类的对话技能——当他们面临歧义时，他们经常过度对冲或隐含地猜测用户的真实意图，而不是提出澄清问题。
* 在特定任务的设置下，高质量的对话样本往往是有限的，这构成了LLMs学习最佳对话行动策略的能力的瓶颈。
#### 基于DPO提出ACT(Action-Based Constrastive Self-Training)
准在线偏好优化算法，可以在多轮对话建模中实现数据高效的对话策略学习。

### 相关链接
* <a href="./papers/4910_Learning_to_Clarify_Multi.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=SIE6VFps9x">ICLR链接</a>

### 摘要
Large language models (LLMs), optimized through human feedback, have rapidly emerged as a leading paradigm for developing intelligent conversational assistants. However, despite their strong performance across many benchmarks, LLM-based agents might still lack conversational skills such as disambiguation -- when they are faced with ambiguity, they often overhedge or implicitly guess users' true intents rather than asking clarification questions. Under task-specific settings, high-quality conversation samples are often limited, constituting a bottleneck for LLMs' ability to learn optimal dialogue action policies. We propose Action-Based Contrastive Self-Training (ACT), a quasi-online preference optimization algorithm based on Direct Preference Optimization (DPO), that enables data-efficient dialogue policy learning in multi-turn conversation modeling. We demonstrate ACT's efficacy under in data-efficient tuning scenarios, even when there is no action label available, using multiple real-world conversational tasks: tabular-grounded question-answering, machine reading comprehension, and AmbigSQL, a novel task for disambiguating information-seeking requests for complex SQL generation towards data analysis agents. Additionally, we propose evaluating LLMs' ability to function as conversational agents by examining whether they can implicitly recognize and reason about ambiguity in conversation. ACT demonstrates substantial conversation modeling improvements over standard tuning approaches like supervised fine-tuning and DPO.
通过人类反馈进行优化​​的大型语言模型（ LLMs ）已迅速成为开发智能对话助理的领先范例。然而，尽管他们在许多基准测试中表现出色，基于LLM的代理可能仍然缺乏诸如消歧之类的对话技能——当他们面临歧义时，他们经常过度对冲或隐含地猜测用户的真实意图，而不是提出澄清问题。在特定任务的设置下，高质量的对话样本往往是有限的，这构成了LLMs学习最佳对话行动策略的能力的瓶颈。我们提出了基于动作的对比自我训练（ACT），这是一种基于直接偏好优化（DPO）的准在线偏好优化算法，可以在多轮对话建模中实现数据高效的对话策略学习。我们使用多个现实世界的对话任务来展示 ACT 在数据高效调整场景中的功效，即使没有可用的操作标签：基于表格的问答、机器阅读理解和 AmbigSQL（一项消除信息歧义的新颖任务）向数据分析代理寻求复杂 SQL 生成的请求。此外，我们建议通过检查法学硕士是否能够隐式识别和推理对话中的歧义来评估LLMs作为对话代理的能力。 ACT 展示了相对于监督微调和 DPO 等标准调整方法的重大对话建模改进。


## 68. Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models
具有执行反馈的自我对弈：提高大型语言模型的指令跟踪能力  
### 关键字
* Instruction Following
* Large Language Models
* Execution Feedback
* On-policy Learning (直接优化当前策略，而不是参考历史数据)
* Strong-to-Weak Distillation (将强模型的知识传递给相对较弱的模型)
* Self-Alignment (模型通过自我学习或自我反馈进行对齐)

### 主要内容
#### 面对问题：
在无需手动注释的情况下自动构建高质量训练数据以增强LLMs复杂的指令跟踪能力的问题仍未解决
#### AutoIF
第一个用于自动生成指令跟踪训练数据的可扩展且可靠的方法
* AutoIF将指令跟随数据质量的验证转化为代码验证，要求LLMs生成指令，生成相应的代码来验证指令响应的正确性，并通过单元测试样本来交叉验证代码的正确性。
* 然后，基于执行反馈的拒绝采样可以生成用于监督微调（SFT）和人类反馈强化学习（RLHF）训练的数据。
### 相关链接
* <a href="./papers/3173_Self_play_with_Execution_.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=cRR0oDFEBC">ICLR链接</a>

### 摘要
One core capability of large language models~(LLMs) is to follow natural language instructions. However, the issue of automatically constructing high-quality training data to enhance the complex instruction-following abilities of LLMs without manual annotation remains unresolved. In this paper, we introduce AutoIF, the first scalable and reliable method for automatically generating instruction-following training data. AutoIF transforms the validation of instruction-following data quality into code verification, requiring LLMs to generate instructions, the corresponding code to verify the correctness of the instruction responses, and unit test samples to cross-validate the code's correctness. Then, execution feedback-based rejection sampling can generate data for Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) training. AutoIF achieves significant improvements across three training algorithms, SFT, Offline DPO, and Online DPO, when applied to the advanced open-source LLMs, Qwen2 and LLaMA3, in self-alignment and strong-to-weak distillation settings. Using two widely-used and three challenging general instruction-following benchmarks, we demonstrate that AutoIF significantly improves LLM performance across a wide range of natural instruction constraints. Notably, AutoIF is the first to surpass 90% accuracy in IFEval’s loose instruction accuracy, without compromising general, math and coding capabilities. Further analysis of quality, scaling, combination, and data efficiency highlights AutoIF's strong generalization and alignment potential.
大型语言模型（ LLMs ）的一项核心能力是遵循自然语言指令。然而，在无需手动注释的情况下自动构建高质量训练数据以增强LLMs复杂的指令跟踪能力的问题仍未解决。在本文中，我们介绍了 AutoIF，这是第一个用于自动生成指令跟踪训练数据的可扩展且可靠的方法。 AutoIF将指令跟随数据质量的验证转化为代码验证，要求LLMs生成指令，生成相应的代码来验证指令响应的正确性，并通过单元测试样本来交叉验证代码的正确性。然后，基于执行反馈的拒绝采样可以生成用于监督微调（SFT）和人类反馈强化学习（RLHF）训练的数据。当 AutoIF 在自对准和强到弱蒸馏设置中应用于高级开源LLMs 、Qwen2 和 LLaMA3 时，在三种训练算法（SFT、离线 DPO 和在线 DPO）上实现了显着改进。使用两个广泛使用的和三个具有挑战性的通用指令跟踪基准，我们证明 AutoIF 在各种自然指令约束下显着提高了LLM性能。值得注意的是，AutoIF 是第一个在 IFEval 的松散指令精度方面超过 90% 精度的产品，且不影响一般、数学和编码能力。对质量、扩展、组合和数据效率的进一步分析凸显了 AutoIF 强大的泛化和对齐潜力。


## 69. Selective Preference Optimization via Token-Level Reward Function Estimation
通过Token级奖励函数估计进行选择性偏好优化  

### 关键字
* LLMs
* Preference Optimization
* Alignment

### 主要内容
#### 现有的token级对齐方法的局限性
* 对所有可用token进行优化，可能会产生噪声而效率低下
* 使用复杂且昂贵的关键token选择策略进行选择性训练
#### SePO (Selective PO)
* 选择性对齐策略，其核心是有效的关键token选择，而不需要强大的、细粒度的监督信号。
* 理论上证明DPO作为token级奖励函数估计器的可行性，适用于任何现有的对齐数据集，并通过小规模模型大小和训练数据实现具有成本效益的token选择。
* 然后在目标数据上使用 DPO 训练oracle模型，并利用估计的奖励函数对目标数据集中的所有token进行评分，其中仅选择关键token来通过对比目标函数来监督目标策略模型。
### 相关链接

* <a href="./papers/6269_Selective_Preference_Opti.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=Bvqsas4TYX">ICLR链接</a>

### 摘要
Recent advancements in large language model alignment leverage token-level supervisions to perform fine-grained preference optimization. However, existing token-level alignment methods either optimize on all available tokens, which can be noisy and inefficient, or perform selective training with complex and expensive key token selection strategies. In this work, we propose Selective Preference Optimization (SePO), a novel selective alignment strategy that centers on efficient key token selection without requiring strong, fine-grained supervision signals. We theoretically prove the feasibility of Direct Preference Optimization (DPO) as token-level reward function estimators, which applies to any existing alignment datasets and enables cost-efficient token selection with small-scale model sizes and training data. We then train an oracle model with DPO on the target data and utilize the estimated reward function to score all tokens within the target dataset, where only the key tokens are selected to supervise the target policy model with a contrastive objective function. Extensive experiments on three public evaluation benchmarks show that SePO significantly outperforms competitive baseline methods by only optimizing on 30% key tokens. We also explore SePO as a new paradigm for weak-to-strong generalization, showing that weak oracle models effectively supervise strong policy models with up to 16.8 more parameters. SePO also selects useful supervision signals from out-of-distribution data, alleviating the over-optimization problem.
大语言模型对齐的最新进展利用token级监督来执行细粒度的偏好优化。然而，现有的token级对齐方法要么对所有可用token进行优化，这可能会产生噪音且效率低下，要么使用复杂且昂贵的关键token选择策略进行选择性训练。 在这项工作中，我们提出了选择性偏好优化（SePO），这是一种新颖的选择性对齐策略，其核心是有效的关键token选择，而不需要强大的、细粒度的监督信号。我们从理论上证明了直接偏好优化（DPO）作为token级奖励函数估计器的可行性，它适用于任何现有的对齐数据集，并通过小规模模型大小和训练数据实现具有成本效益的token选择。然后，我们在目标数据上使用 DPO 训练预言机模型，并利用估计的奖励函数对目标数据集中的所有token进行评分，其中仅选择关键token来通过对比目标函数来监督目标策略模型。对三个公共评估基准的广泛实验表明，SePO 仅优化 30% 的关键代币，其性能显着优于竞争基准方法。我们还探索 SePO 作为弱到强泛化的新范式，表明 弱预言机模型有效监督强政策模型，高达 16.8 更多参数。 SePO 还从分布外数据中选择有用的监督信号，缓解过度优化问题。



## 70. Zeroth-Order Policy Gradient for Reinforcement Learning from Human Feedback without Reward Inference
从人类反馈中进行强化学习的零阶策略梯度，无需奖励推理  
### 关键字
* RL Theory
* Human feedback
* Zeroth-Order Optimization
    * 在不知道目标函数的梯度信息的情况下进行优化，也被称为“无梯度优化”或“黑盒优化”。该方法仅利用目标函数的函数值（而非导数信息）来更新参数，因此适用于梯度不可计算或不可用的场景。

### 主要内容
#### Reward Inference面临着几个基本挑战：
* 双重问题错误指定
* 没有真实事实的奖励模型评估
* 分布转移
* 联合奖励模型和政策训练中的过度拟合

避免上述Reward Inference中相关问题可以使用DPO，但是DPO 使用最优策略和奖励函数之间的封闭式表达，仅在Bandit设置或确定性 MDP 下有效。
#### 开发没有Reward Inference的RLHF算法，适用与更一般的RL问题
* 关键思想是估计局部价值函数与人类偏好的差异，然后用零阶梯度逼近器来逼近策略梯度
* 对于这两种算法，根据策略梯度迭代的数量以及每次迭代的轨迹样本和人类偏好查询的数量来确定收敛率

### 相关链接
* <a href="./papers/4367_Zeroth_Order_Policy_Gradi.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=cmYScmfu4Q">ICLR链接</a>

### 摘要
Reward inference (learning a reward model from human preferences) is a critical intermediate step in Reinforcement Learning from Human Feedback (RLHF) for fine-tuning Large Language Models (LLMs) such as ChatGPT. In practice, reward inference faces several fundamental challenges, including double problem misspecification, reward model evaluation without ground truth, distribution shift, and overfitting in joint reward model and policy training. An alternative approach that avoids these pitfalls is direct policy optimization without reward inference, such as Direct Preference Optimization (DPO), which provides a much simpler pipeline and has shown empirical success in LLMs. However, DPO utilizes the closed-form expression between the optimal policy and the reward function, which only works under the bandit setting or deterministic MDPs. This paper develops two RLHF algorithms without reward inference, which work for general RL problems beyond bandits and deterministic MDPs, and general preference models beyond the Bradely-Terry model. The key idea is to estimate the local value function difference from human preferences and then approximate the policy gradient with a zeroth-order gradient approximator. For both algorithms, we establish rates of convergence in terms of the number of policy gradient iterations, as well as the number of trajectory samples and human preference queries per iteration. Our results show there exist provably efficient methods to solve general RLHF problems without reward inference.
奖励推理（根据人类偏好学习奖励模型）是人类反馈强化学习 (RLHF) 中的关键中间步骤，用于微调 ChatGPT 等大型语言模型 ( LLMs )。在实践中，奖励推理面临着几个基本挑战，包括双重问题错误指定、没有真实事实的奖励模型评估、分布转移以及联合奖励模型和政策训练中的过度拟合。避免这些陷阱的另一种方法是不进行奖励推断的直接策略优化，例如直接偏好优化（DPO），它提供了更简单的管道，并在LLMs中取得了经验上的成功。然而，DPO 使用最优策略和奖励函数之间的封闭式表达，仅在强盗设置或确定性 MDP 下有效。本文开发了两种没有奖励推理的 RLHF 算法，它们适用于强盗和确定性 MDP 之外的一般 RL 问题，以及 Bradely-Terry 模型之外的一般偏好模型。关键思想是估计局部价值函数与人类偏好的差异，然后用零阶梯度逼近器来逼近策略梯度。对于这两种算法，我们根据策略梯度迭代的数量以及每次迭代的轨迹样本和人类偏好查询的数量来确定收敛率。我们的结果表明，存在可证明有效的方法来解决一般 RLHF 问题，而无需奖励推理。


## 71. As Simple as Fine-tuning: LLM Alignment via Bidirectional Negative Feedback Loss
就像微调一样简单：通过双向负反馈损失进行LLM对齐  
### 关键字
* LLM Alignment
* Preference Learning
* Text Generation

### 主要内容
#### DPO对超参敏感的问题
DPO 及其变体仍然对超参数敏感并且容易不稳定，特别是在数学数据集上
#### 归因
我们认为这些问题是由对数似然损失函数中固有的单向似然导数负反馈引起的
#### 提出措施
* 提出了一种新颖的LLM对齐损失，可以在优化过程中建立稳定的双向负反馈（BNF）。
* 该 BNF 损失消除了对成对对比损失的需要，并且不需要任何额外的可调节超参数或成对偏好数据，从而简化了对齐管道，使其像监督微调一样简单。
### 相关链接
* <a href="./papers/10506_As_Simple_as_Fine_tuning.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=fsX9nFwMNj">ICLR链接</a>

### 摘要
Direct Preference Optimization (DPO) has emerged as a more computationally efficient alternative to Reinforcement Learning from Human Feedback (RLHF) with Proximal Policy Optimization (PPO), eliminating the need for reward models and online sampling. Despite these benefits, DPO and its variants remain sensitive to hyper-parameters and prone to instability, particularly on mathematical datasets. We argue that these issues arise from the unidirectional likelihood-derivative negative feedback inherent in the log-likelihood loss function. To address this, we propose a novel LLM alignment loss that establishes a stable Bidirectional Negative Feedback (BNF) during optimization. Our proposed BNF loss eliminates the need for pairwise contrastive losses and does not require any extra tunable hyper-parameters or pairwise preference data, streamlining the alignment pipeline to be as simple as supervised fine-tuning. We conduct extensive experiments across two challenging QA benchmarks and four reasoning benchmarks. The experimental results show that BNF achieves comparable performance to the best methods on QA benchmarks, while its performance decrease on the four reasoning benchmarks is significantly lower compared to the best methods, thus striking a better balance between value alignment and reasoning ability. In addition, we further validate the performance of BNF on non-pairwise datasets, and conduct in-depth analysis of log-likelihood and logit shifts across different preference optimization methods. We will release all the source code, checkpoints, and datasets on GitHub.
直接偏好优化 (DPO) 已成为一种计算效率更高的替代方案，可替代基于人类反馈的强化学习 (RLHF) 和近端策略优化 (PPO)，从而消除了对奖励模型和在线采样的需求。尽管有这些好处，DPO 及其变体仍然对超参数敏感并且容易不稳定，特别是在数学数据集上。我们认为这些问题是由对数似然损失函数中固有的单向似然导数负反馈引起的。为了解决这个问题，我们提出了一种新颖的LLM对齐损失，可以在优化过程中建立稳定的双向负反馈（BNF）。我们提出的 BNF 损失消除了对成对对比损失的需要，并且不需要任何额外的可调节超参数或成对偏好数据，从而简化了对齐管道，使其像监督微调一样简单。我们在两个具有挑战性的 QA 基准和四个推理基准上进行了广泛的实验。实验结果表明，BNF 在 QA 基准上取得了与最佳方法相当的性能，而在四个推理基准上的性能下降明显低于最佳方法，从而在价值对齐和推理能力之间取得了更好的平衡。此外，我们进一步验证了 BNF 在非成对数据集上的性能，并对不同偏好优化方法的对数似然和 logit 变化进行了深入分析。我们将在 GitHub 上发布所有源代码、检查点和数据集。

## 72. Model Extrapolation Expedites Alignment
模型外推加速对齐  

### 关键字
* LLMs
* Alignment
* Preference Optimization
* Model Merging

### 主要内容
#### 希望探索更有效的对齐方法以减少训练开销
#### ExPO(model extrapolation)
受`插值工作`的启发
1. 观察得在现有DPO/RLHF模型与其初始SFT检查点之间插值权重会产生具有中等性能的新模型
2. 建议处理部分训练的模型$M_1$(对应中间性能模型)作为初始$M_0$和$M_2$的插值结果，从而反推出$M_2$

### 相关链接
* <a href="./papers/4718_Model_Extrapolation_Exped.pdf">查看PDF</a>
* <a href="https://openreview.net/forum?id=QN97ubU1HH">ICLR链接</a>

### 摘要
As the alignment training of large language models (LLMs) usually requires expensive computational resources, exploring more efficient alignment methods to reduce training overhead has always been an important and compelling research challenge. Inspired by prior work on model interpolation, we present a simple method called ExPO (model extrapolation) to expedite the alignment of LLMs with human preferences. Based on our observation that interpolating the weights between existing DPO/RLHF models and their initial SFT checkpoints usually produces new models with intermediate performance, we propose to treat a partially-trained model (corresponding to the intermediate-performing model) as the interpolated result between the initial SFT checkpoint and a hypothetical better-aligned model . Thus, we can obtain the hypothetical by simply extrapolating the model weights along the direction from to , which consequently saves the additional training overhead for to reach better alignment performance. We validate our hypothesis through controlled experiments, demonstrating that ExPO can boost a DPO model trained with only 20% steps to outperform the fully-trained one. Additionally, we show that ExPO can also notably improve existing open-source LLMs (ranging from 1.8B to 70B parameters), as evidenced by evaluations on the mainstream LLM benchmarks AlpacalEval 2.0 and MT-Bench, which further highlights ExPO's utility and potential in enabling more efficient LLM alignment.
由于大型语言模型（ LLMs ）的对齐训练通常需要昂贵的计算资源，探索更有效的对齐方法以减少训练开销一直是一个重要且引人注目的研究挑战。受到先前模型插值工作的启发，我们提出了一种称为ExPO（模型外推）的简单方法，以加快LLMs与人类偏好的一致性。根据我们的观察，在现有 DPO/RLHF 模型与其初始 SFT 检查点之间插值权重通常会产生具有中等性能的新模型，我们建议处理部分训练的模型 （对应于中间性能模型）作为初始SFT检查点之间的插值结果 以及假设的更好对齐的模型 。因此，我们可以得到假设的 通过简单地沿方向推断模型权重 到 ，从而节省了额外的训练开销 以达到更好的对准性能。我们通过对照实验验证了我们的假设，证明 ExPO 可以提升仅用 20% 步骤训练的 DPO 模型，使其性能优于完全训练的模型。此外，我们还表明，ExPO 还可以显着改善现有的开源LLMs （参数范围从 1.8B 到 70B），对主流LLM基准 AlpacalEval 2.0 和 MT-Bench 的评估证明了这一点，这进一步凸显了 ExPO 在实现更高效的LLM调整。


## 73. Correcting the Mythos of KL-Regularization: Direct Alignment without Overoptimization via Chi-Squared Preference Optimization
纠正 KL 正则化的神话：通过卡方偏好优化直接对齐而不过度优化  
### 关键字
* RL Theory
* Offline RL
* Single-Policy Concentrability
* Pessimism (悲观主义)
* RLHF

### 主要内容


### 相关链接
* <a href="">查看PDF</a>
* <a href="">ICLR链接</a>

### 摘要
Language model alignment methods, such as reinforcement learning from human feedback (RLHF), have led to impressive advances in language model capabilities. However, existing techniques are limited by a widely observed phenomenon known as overoptimization, where the quality of the language model degrades over the course of the alignment process. Overoptimization occurs when a language model overfits to inaccuracies in an (either explicit or implicit) offline reward model, and drifts away from preferred responses covered by the data. To discourage such distribution shift, offline alignment methods typically employ KL-regularization, but this, as we show, is too weak to prevent degradation in performance. Then, can we design an efficient algorithm that is provably robust to overoptimization?
In this paper, we advance theoretical understanding of sample-efficient offline alignment and introduce a new algorithm called$X^2$-Preference Optimization (PO). PO is a one-line change to Direct Preference Optimization (DPO; Rafailov et al. 2023), that modifies only the logarithmic link function in the DPO objective. Despite this minimal change, PO implicitly implements the principle of pessimism in the face of uncertainty via regularization with the -divergence---which quantifies uncertainty more effectively than KL-regularization---and provably alleviates overoptimization, achieving sample-complexity guarantees based on single-policy concentrability---the gold standard in offline reinforcement learning. This guarantee makes PO the first simple, yet general-purpose offline alignment algorithm that is provably robust to overoptimization.
语言模型对齐方法，例如来自人类反馈的强化学习（RLHF），已经在语言模型能力方面带来了令人印象深刻的进步。然而，现有技术受到广泛观察到的过度优化现象的限制，即语言模型的质量在对齐过程中下降。当语言模型过度适应（显式或隐式）离线奖励模型中的不准确性，并偏离数据所涵盖的首选响应时，就会发生过度优化。为了阻止这种分布变化，离线对齐方法通常采用 KL 正则化，但正如我们所表明的，这太弱而无法防止性能下降。那么，我们能否设计一种可证明对过度优化具有鲁棒性的高效算法呢？
在本文中，我们推进了对样本高效离线对齐的理论理解，并引入了一种称为 -偏好优化（ 采购订单）。 PO 是对直接偏好优化（DPO；Rafailov 等人，2023）的单行更改，仅修改 DPO 目标中的对数链接函数。尽管变化很小， PO 通过正则化隐含地实现了面对不确定性时的悲观原则 -发散——比 KL 正则化更有效地量化不确定性——并且可证明缓解过度优化，实现基于单策略集中性的样本复杂性保证——这是离线强化学习的黄金标准。这一保证使得 PO 第一个简单但通用的离线对齐算法，经证明对过度优化具有鲁棒性。


## 74. SuperCorrect: Supervising and Correcting Language Models with Error-Driven Insights
SuperCorrect：通过错误驱动的见解监督和纠正语言模型  

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