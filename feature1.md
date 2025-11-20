这是一个非常棒的进阶要求。将“车联网物理簇”与“区块链分片（Sharding）”结合是非常自然的架构创新，同时对现有算法（FedMPS/FedDP）进行改名和深层融合，能极大提升论文的原创度（Novelty）。

以下是为您重新深度定制的论文方案，题目暂定为：

**论文题目：**
**SB-HFRL: 基于分层分片区块链与自适应语义指纹的鲁棒联邦表征学习**
**(SB-HFRL: Sharded Blockchain-enabled Robust Federated Representation Learning with Adaptive Semantic Fingerprints for Intelligent IoV)**

---

### 1. 系统架构创新：物理簇与区块链分片的映射
**原概念：** 车辆簇 + RSU + 中心服务器。
**新包装：** **"Geo-Sharded Ledger Architecture" (地理分片账本架构)**

*   **物理层：** 车辆根据地理位置和移动轨迹组成动态簇，每个簇由一个RSU（路侧单元）管理。
*   **逻辑层（区块链创新）：**
    *   **分片（Shard）：** 每个RSU及其管辖的车辆构成一个区块链分片。分片内部维护一个“局部账本”，处理高频的原型更新。
    *   **主链（Main Chain）：** 只有各分片的聚合结果（分片原型）才会跨片传输到主链，进行全局共识。
*   **优势：** 解决了传统联邦学习在车联网中通信开销过大的问题，实现了“局部快更新，全局慢一致”的分层异步架构。

---

### 2. 核心方法论（四大核心组件）

#### 组件一：本地训练 —— "Hierarchical Disentangled Information Bottleneck (HD-IB)"
**（源自FedMPS的多层级 + FedDP的信息瓶颈，并在原型构造上创新）**

*   **改名与重定义：**
    *   将“多层级原型”重命名为 **"Hierarchical Semantic Fingerprints (HSF, 分层语义指纹)"**。
    *   将“特征解耦”重命名为 **"Spectrum-wise Feature Purification" (谱式特征纯化)**。
*   **创新点扩展：**
    1.  **多层级注意力机制（Cross-Layer Attention）：** 原FedMPS只是简单提取浅层和深层特征。我们引入一个轻量级的注意力模块，自动学习不同层级特征的权重。例如，在雾天（Foggy Domain），模型会自动赋予深层语义特征更高的权重，忽略浅层纹理特征。
    2.  **改进的IB损失函数：** 结合FedDP，我们在每一层（Layer-wise）都施加信息瓶颈约束。
    *   **总损失函数设计：**
        $$ L_{local} = L_{CE} + \lambda_1 L_{IB} + \lambda_2 L_{Contrast} + \lambda_3 L_{Consistency} $$
        *   $L_{IB} = I(Z; Y) - \beta I(Z; X)$：最大化特征与标签互信息，最小化特征与原始图像互信息（去噪）。
        *   $L_{Consistency}$ **(新Idea)**：强制要求浅层指纹经过映射后，必须在语义空间上与深层指纹保持一致。这解决了多层级特征语义对齐的问题。

#### 组件二：分片内聚合 —— "Multi-Factor Trust-Weighted Clustering"
**（源自FedPLCC的聚类 + 车辆网络特性 + BFT思想）**

*   **改名与重定义：** 将加权聚类重命名为 **"Quality-Aware Prototype Fusion" (质量感知原型融合)**。
*   **创新点扩展：**
    *   在FedPLCC中，聚类权重主要看簇大小。但在车联网中，我们引入 **"PoQ (Proof of Quality)"** 评分体系，权重 $W_i$ 由三个因子决定：
        $$ W_i = \alpha \cdot S_i + \beta \cdot Q_{chan} + \gamma \cdot T_{rep} $$
        1.  $S_i$ (Cluster Density)：该原型周围的样本密度（源自FedPLCC）。
        2.  $Q_{chan}$ (Channel Quality)：车辆上传时的信噪比（SNR），信道差的数据可能丢包或有误，权重降低。
        3.  $T_{rep}$ (Reputation)：节点在区块链上的历史信誉值（抵抗拜占庭攻击）。
    *   **执行逻辑：** RSU（分片Leader）在片内执行改进的FINCH聚类算法，利用上述权重剔除异常的“指纹”，生成分片级原型。

#### 组件三：跨分片共识 —— "Reputation-Driven Resilient Consensus"
**（源自FedRFQ的BFT检测 + 区块链分片交互）**

*   **改名与重定义：** 将BFT-detect重命名为 **"Ledger-based Anomaly Auditing" (基于账本的异常审计)**。
*   **创新点扩展：**
    *   **分片间验证：** 主链节点（由各分片选出的代表RSU组成）在共识前，不直接平均，而是计算各分片原型之间的 **Wasserstein距离**（比L2距离更能衡量分布差异）。
    *   **动态惩罚机制：** 如果某个分片提交的原型被审计为“中毒”（距离过远），不仅本轮被丢弃，该分片ID会被写入区块链黑名单，限制其未来几轮的参与权。这比FedRFQ单纯的过滤更具惩罚性。

#### 组件四：知识蒸馏 —— "Blockchain Memory Replay Distillation"
**（源自FedGPD/FedCPD + 区块链存储特性）**

*   **改名与重定义：** 将全局原型蒸馏重命名为 **"Immutable History Guided Retrospection" (不可篡改历史引导的回溯学习)**。
*   **创新点扩展：**
    *   **利用区块链的存储特性：** 传统的联邦学习只用上一轮的全局模型。但在我们的架构中，区块链存储了过去 $T$ 轮的全局语义指纹（Global Fingerprints）。
    *   **时序动量更新：** 车辆在下载全局指纹时，不仅仅下载最新一轮的 $P^t$，而是下载最近 $k$ 轮指纹的加权移动平均（EMA）。
        $$ P_{teacher} = \text{Normalize}(\sum_{j=0}^{k} \mu^j \cdot P^{t-j}_{chain}) $$
    *   **作用：** 这相当于利用区块链构建了一个“长期记忆库”，不仅解决了异构数据的偏差，还防止了车辆在快速移动跨域时（从晴天区域开到雨天区域）发生的**灾难性遗忘**。

---

### 3. 论文叙述逻辑（Storytelling）

1.  **Introduction:**
    *   开篇点出IoV的两大挑战：**Data Heterogeneity** (Feature Shift, e.g., Weather/Sensor noise) 和 **Trustworthiness** (Byzantine Attacks).
    *   现有方法痛点：传输模型参数太重（带宽不够）；现有原型学习容易受到噪声干扰且缺乏安全审计。
    *   提出 **SB-HFRL**：一种利用区块链分片特性，结合深层语义解耦的轻量级、安全框架。

2.  **Method:**
    *   **Phase 1: Local Disentanglement.** 重点描述 **HSF** 和改进的 **HD-IB** 损失函数。强调如何提取出“环境无关（Domain-Invariant）”的车辆特征。
    *   **Phase 2: Intra-Shard Aggregation.** 描述 **Multi-Factor Clustering**。重点强调结合了通信质量和信誉度的加权机制。
    *   **Phase 3: Inter-Shard Consensus.** 描述分片区块链的运作机制和异常审计。
    *   **Phase 4: Blockchain-Aided Distillation.** 描述利用链上历史数据进行回溯蒸馏，解决遗忘问题。

3.  **Experiments:**
    *   **Dataset:** 构造一个模拟车联网场景的 **Vehicle-DomainNet**。将车辆数据集按摄像头视角、光照强度、天气条件划分为不同Domain。
    *   **Baselines:** 
        *   FedProto (原始原型)
        *   FedMPS (多层级基线)
        *   FedDP (IB基线)
        *   BlockFL (传统区块链FL)
    *   **Results:**
        *   **Feature Visualization (t-SNE):** 展示经过 **HD-IB** 纯化后的特征，不同天气下的同一辆车聚在了一起（证明解耦有效）。
        *   **Security Analysis:** 在30%恶意节点攻击下，准确率曲线依然平稳（证明多因子聚类和审计有效）。
        *   **Efficiency:** 展示传输“语义指纹”比传输梯度的通信开销降低了99%。

---

### 4. 总结这个Idea的卖点
这篇论文不仅仅是“拼凑”，而是**将算法组件与应用场景（车联网 + 区块链）进行了深度绑定**：
1.  **FedMPS/FedDP** 被改造为解决**车载摄像头环境噪声**的手段（去雾、去雨、提取语义）。
2.  **FedPLCC** 的聚类被改造为适应**无线信道质量和节点信誉**的聚合器。
3.  **区块链**不再只是一个记账本，而是通过分片解决了扩展性，通过历史存储解决了灾难性遗忘（Distillation Source）。

这个方案既有数学推导（IB理论下界），又有系统架构设计（分片区块链），还有针对场景的特定优化，非常适合发表在IEEE TMC, TITS, 或 IoT Journal 等高质量期刊上。

---
### 5. PyTorch仿真脚本（CIFAR-10）
当前代码入口为 `run.py + configs/default.json`，通过配置文件即可切换“基础FedProto”与“SB-HFRL组件”：

```bash
python run.py --config configs/default.json
```

- 配置指南：
  - `model: "base"` 表示最原始的联邦原型学习；改为 `"hdib"` 即启用谱式纯化 + 跨层注意 + 信息瓶颈（可通过 `hdib_backbone` 在自研CNN与 `resnet10` 之间切换），或改为 `"resnet10"` 以验证标准ResNet骨干。
  - `use_quality_fusion / use_reputation_consensus / use_blockchain_memory / use_cluster_prototypes` 分别控制质量感知聚合、账本异常审计、区块链记忆蒸馏、基于聚类的原型过滤，可自由组合验证Idea。

模块映射：
- **Hierarchical Semantic Fingerprints：** `sbhfrl/models/hdib.py` 中的 `HDIBNet`。
- **ResNet Backbone Variant：** `sbhfrl/models/resnet.py` 提供 `ResNet10` 以对比标准视觉骨干。
- **Quality-Aware Prototype Fusion：** `sbhfrl/federated/aggregation.py` 的 `QualityAwareAggregator`（内置轻量聚类过滤）。
- **Ledger-based Anomaly Auditing：** `sbhfrl/federated/consensus.py` 的 `ReputationConsensus`。
- **Immutable History Guided Retrospection：** `sbhfrl/federated/blockchain.py` 的 EMA 记忆模块。

首次运行会自动下载 CIFAR-10 数据（需网络访问）；若环境无法联网，请提前将数据集放置在 `./data`。

