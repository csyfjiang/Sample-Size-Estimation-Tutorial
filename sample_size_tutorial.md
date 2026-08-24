# 预测模型样本量估算教程：从开发到验证到模型比较

## 一、背景

临床预测模型研究（如 BC-BioMIXER / BioMIXER v2 / LongiMIXER 系列工作）在每一个阶段都面临一个共同的方法学问题：**手头的样本量，到底够不够回答我想问的问题？**

这个问题在文献中长期依赖"经验法则"来回答——例如"每个预测变量至少需要10个结局事件"（10 EPV）、"外部验证至少需要100个事件和100个非事件"。近年一系列方法学论文（Riley、Snell、van Smeden、Collins、Jung 等人的工作）系统性地指出，这些经验法则**与具体模型、具体临床场景无关**，实际所需的样本量取决于结局比例、候选预测变量数量、模型预期表现（R²/c统计量）、目标精度等多个因素，套用统一的经验法则常常导致样本量过小（估计不精确）或过大（浪费资源）。

## 二、研究问题

围绕一个典型的预测模型研究全流程，存在四类彼此独立但又相互衔接的样本量问题：

1. **开发阶段**：用于训练模型的开发集（如 Cohort 1）样本量是否足以在不过拟合的前提下，得到稳定、可靠的模型参数？
2. **外部验证阶段（单模型精度，闭式解）**：用于评估已训练好的模型在新数据中表现的验证集，其样本量能否让区分度（c统计量）、校准度（校准斜率、O/E比值）、临床效用（净获益）的估计足够精确？
3. **外部验证阶段（单模型精度，模拟法）**：当闭式公式的假设（如线性预测值分布形态）难以满足、或需要同时探究失校准情形时，如何用基于模拟的方法达到同样的目的？
4. **模型间比较阶段**：当研究目标从"这一个模型准不准"转变为"我的新模型是否显著优于某个基线/竞品模型"时，样本量该如何计算？这本质上是"两个相关ROC曲线AUC差异检验"的样本量问题。

## 三、教程目的

本教程配合四个可复用的 Python 脚本（`estimation_ss_1.py` ~ `estimation_ss_4.py`），目标是：

- 讲清楚每个脚本背后的**数学原理**（符号定义、公式推导逐步展开、参数含义）
- 讲清楚每个脚本需要什么样的**输入数据格式**（如果需要从真实数据估计参数，应准备什么样的 CSV，每一列代表什么）
- 讲清楚每种方法适用于**开发流程的哪个阶段**，以及彼此之间的先后逻辑关系
- 提供一份可直接套用的 **manuscript sample-size rationale 段落写作模板**，对应 TRIPOD+AI 条目10的报告要求

## 四、四个脚本的建议阅读/使用顺序

四篇原始方法学论文的发表时间线是：

| 脚本 | 对应论文 | 发表年份 |
|---|---|---|
| `estimation_ss_3` | Riley et al., *BMJ* 2020;368:m441 | 2020 |
| `estimation_ss_2` | Snell et al., *J Clin Epidemiol* 2021;135:79-89 | 2021 |
| `estimation_ss_1` | Riley et al. (part 3), *BMJ* 2023;383:e074821 | 2023 |
| `estimation_ss_4` | Jung, *Pharmaceutical Statistics* 2024;23(4):557-569 | 2024 |

按发表年份排列即为 **`estimation_ss_3`（2020）→ `estimation_ss_2`（2021）→ `estimation_ss_1`（2023）→ `estimation_ss_4`（2024）**。

值得留意的是，这一发表年份顺序恰好也与一次真实研究项目的**逻辑执行顺序**基本吻合：

> **`estimation_ss_3`（开发） → `estimation_ss_2` / `estimation_ss_1`（外部验证，二选一或互证） → `estimation_ss_4`（模型间比较）**

理由：一项预测模型研究天然分为"先开发、后评估、（可选）再比较"三个阶段，这也是 TRIPOD+AI 报告清单本身的组织逻辑（条目10要求分别为开发数据集和评估数据集说明样本量依据）。因此，正文的讲解顺序为：

1. `estimation_ss_3`（开发阶段，2020）
2. `estimation_ss_2`（外部验证，模拟法，2021）
3. `estimation_ss_1`（外部验证，闭式解，2023，与 ss_2 互为印证/备选）
4. `estimation_ss_4`（模型间头对头比较，2024）

---

## 五、`estimation_ss_3`：模型开发阶段样本量

### 5.1 适用阶段
用于回答："我的开发集（Cohort 1）样本量，是否足以支撑一个 P 个候选预测变量的二分类（如 pCR）预测模型的稳定开发？"

### 5.2 数学原理逐步展开

**符号定义：**

| 符号 | 含义 |
|---|---|
| $N$ | 开发集总样本量 |
| $\varphi$ | 目标人群中结局事件（如 pCR）的真实/预期比例 |
| $P$ | 候选预测变量参数数量（注意：不是"变量"数，是"参数"数——一个三分类变量需要2个参数，一个非线性项也需要额外参数） |
| $R^2_{cs}$ | 预期模型表现，Cox-Snell R² |
| $\max(R^2_{cs})$ | 在给定 $\varphi$ 下，Cox-Snell R² 理论上限（详见步骤4） |
| $S$ | 收缩因子（shrinkage factor），理想值1，$S<1$ 表示存在过拟合导致的收缩 |
| $\delta$ | 表观 $R^2_{Nagelkerke}$ 与乐观校正后 $R^2_{Nagelkerke}$ 之间的差值（乐观偏差） |

**步骤1：准则 B1 —— 精确估计总体结局比例（模型截距）**

95% 置信区间近似为：

$$\hat{\varphi} \pm 1.96\sqrt{\frac{\hat{\varphi}(1-\hat{\varphi})}{N}}$$

设目标边际误差为 $\delta_{margin}$（论文建议 $\leq 0.05$），则：

$$N_{B1} = \left(\frac{1.96}{\delta_{margin}}\right)^2 \hat{\varphi}(1-\hat{\varphi})$$

反查模式（已知 $N$，求实际达成的边际误差）：

$$\delta_{margin,achieved} = 1.96\sqrt{\frac{\varphi(1-\varphi)}{N}}$$

**步骤2：准则 B2 —— 平均绝对预测误差（MAPE），仅适用于 $P \leq 30$**

van Smeden et al. 的经验公式：

$$\ln(\text{MAPE}) = -0.508 - 0.544\ln(N) + 0.259\ln(\varphi) + 0.504\ln(P)$$

（若 $\varphi > 0.5$，公式中 $\varphi$ 替换为 $1-\varphi$）

反查（已知 $N$，求达成的 MAPE）：

$$\text{MAPE}_{achieved} = \exp\left[-0.508 - 0.544\ln(N) + 0.259\ln(\varphi) + 0.504\ln(P)\right]$$

前瞻求解（已知目标 MAPE，求所需 $N$）：

$$N_{B2} = \exp\left[\frac{-0.508 + 0.259\ln(\varphi) + 0.504\ln(P) - \ln(\text{MAPE}_{target})}{0.544}\right]$$

**步骤3：准则 B3 —— 收缩因子（通常是瓶颈项）**

Riley et al. 推导的闭式关系：

$$N = \frac{P}{(S-1)\ln\left(1 - \dfrac{R^2_{cs}}{S}\right)}$$

前瞻求解（已知目标 $S$，通常取 $S \geq 0.9$，即收缩 $\leq 10\%$）：

$$N_{B3} = \frac{P}{(S_{target}-1)\ln\left(1 - \dfrac{R^2_{cs}}{S_{target}}\right)}$$

反查（已知 $N$，反解 $S$）：由于该方程无法解析地反解出 $S$，脚本内部使用 `scipy.optimize.brentq` 在 $S \in (R^2_{cs}, 0.999999)$ 区间内做数值求根，寻找满足下式的 $S$：

$$f(S) = \frac{P}{(S-1)\ln\left(1 - \dfrac{R^2_{cs}}{S}\right)} - N = 0$$

**步骤4：$\max(R^2_{cs})$ 的计算**（二分类结局特有）

$$L_0^{1/N} = \varphi^{\varphi}(1-\varphi)^{1-\varphi}$$

$$\max(R^2_{cs}) = 1 - \left[\varphi^{\varphi}(1-\varphi)^{1-\varphi}\right]^2$$

**步骤5：准则 B4 —— 表观 $R^2_{Nagelkerke}$ 的乐观偏差**

前瞻求解（已知目标乐观偏差 $\delta_{target}$，通常 $\leq 0.05$）：

$$S_{B4} = \frac{R^2_{cs}}{R^2_{cs} + \delta_{target} \cdot \max(R^2_{cs})}$$

再代入步骤3的公式反推 $N_{B4}$。

反查（已知 $N$，先用步骤3反解出 $\hat{S}$，再反推乐观偏差）：

$$\hat{\delta} = \frac{R^2_{cs}(1-\hat{S})}{\hat{S} \cdot \max(R^2_{cs})}$$

**步骤6：取四者最大值**

$$N_{required} = \max(N_{B1}, N_{B2}, N_{B3}, N_{B4})$$

### 5.3 需要的数据/输入

本脚本**不需要外部 CSV 文件**——它是纯参数化计算，所有输入都是标量：

| 参数名（脚本变量） | 含义 | 来源建议 |
|---|---|---|
| `PHI` | 目标人群结局比例 $\varphi$ | 你的开发队列中实际的 pCR 阳性率，或既往文献报告值 |
| `P` | 候选预测变量参数数量 | 需要你自行界定：对于回归模型是变量数（含哑变量、非线性项展开后的参数数）；对于 BC-BioMIXER 这类融合模型，建议使用保守代理指标（如参与融合的独立特征通道数），而非神经网络权重总数 |
| `R2CS_GUESS` | 预期 Cox-Snell R² | 若无先验信息，用默认保守规则 $R^2_{cs} = 0.15 \times \max(R^2_{cs})$；若已有既往同类模型报告的 c 统计量，可换算（见第八节） |
| `cohort1_n` | 你的开发集实际样本量 | 直接填入，如 717 |

若你希望批量对多组假设做敏感性分析（如同时测试 $P=10, 20, 30$），可以扩展脚本读取一个简单的**假设组合表**，建议格式：

```
scenario_id,phi,P,r2cs_guess,cohort_n
1,0.30,20,0.106,717
2,0.30,10,0.106,717
3,0.30,20,0.212,717
```

（列名：`scenario_id`=情景编号；`phi`=结局比例；`P`=候选参数数；`r2cs_guess`=预期R²cs；`cohort_n`=开发集样本量）

---

## 六、`estimation_ss_2`：外部验证样本量（基于模拟，二分类结局）

### 6.1 适用阶段
模型已经训练好，现在要评估它在一个新数据集（如 Cohort 3、Cohort 4）中的表现精度是否足够可靠。本方法采用**模拟法**而非闭式公式，优势是可以直接指定任意线性预测值分布形态，并能显式模拟"失校准"情形。

### 6.2 数学原理逐步展开

**符号定义：**

| 符号 | 含义 |
|---|---|
| $LP_i$ | 第 $i$ 位患者的模型线性预测值（对数几率尺度） |
| $\mu, \sigma$ | 假设 $LP_i \sim N(\mu, \sigma^2)$ 的均值与标准差 |
| $\gamma$（脚本中记为 `gamma`） | 真实校准模型的截距（calibration-in-the-large），对应 CITL |
| $S$ | 真实校准模型的斜率（脚本中记为 `S`），$S=1$ 表示良好校准 |

**步骤1：设定验证人群的线性预测值分布**

$$LP_i \sim N(\mu, \sigma^2)$$

**步骤2：设定真实校准模型（用于生成"标签"）**

$$\text{logit}(p_i^{true}) = \gamma + S \cdot LP_i$$

$$p_i^{true} = \frac{1}{1+\exp\left[-(\gamma + S \cdot LP_i)\right]}$$

$$Y_i \sim \text{Bernoulli}(p_i^{true})$$

**步骤3：模型自身给出的预测概率**（假设模型按其自身设定直接输出，即 $\alpha=0, \lambda=1$ 应用于 $LP_i$）：

$$\hat{p}_i = \frac{1}{1+\exp(-LP_i)}$$

**步骤4：重复步骤1-3共 $B$ 次**（论文建议 $B=500$，脚本默认可设为500或更多），每次模拟出一个验证数据集，在该数据集上估计：

- 校准模型：拟合 $\text{logit}(\hat{p}_i^{fit}) = \hat{\gamma} + \hat{S} \cdot LP_i$（Newton-Raphson 迭代逻辑回归，脚本内部实现），得到 $\hat{\gamma}, \hat{S}$ 及其标准误
- 总体校准度（CITL，固定斜率为1的偏移模型）：拟合 $\text{logit}(\hat{p}_i^{fit}) = \hat{\gamma}_{CITL} + 1 \cdot LP_i$
- c 统计量：通过 Mann-Whitney U 统计量计算，标准误用 Hanley-McNeil 近似公式：

$$SE(C) = \sqrt{\frac{C(1-C) + (n_1-1)(q_1-C^2) + (n_0-1)(q_2-C^2)}{n_1 n_0}}$$

其中 $q_1 = \dfrac{C}{2-C}$，$q_2 = \dfrac{2C^2}{1+C}$，$n_1, n_0$ 分别是有/无结局的患者数

- O/E 比值：$O/E = \dfrac{\sum Y_i}{\sum \hat{p}_i}$
- 净获益（在阈值 $p_t$ 处）：

$$NB_{p_t} = sensitivity \times prevalence - (1-specificity) \times (1-prevalence) \times \frac{p_t}{1-p_t}$$

**步骤5：汇总 $B$ 次模拟的平均95%置信区间宽度**，作为该 $N$ 下各指标"预期能达到的精度"。

**步骤6（前瞻模式，`forward_search_sample_size` 函数）**：从某个起始 $N$ 开始，每次增加固定步长，重复步骤1-5，直至所有目标精度都达标为止。

### 6.3 需要的数据/输入

同样是纯参数化模拟，不需要外部 CSV。但若你想用**你实际模型的线性预测值经验分布**替代正态假设（更贴近真实情况），建议准备如下 CSV：

| 列名 | 含义 |
|---|---|
| `patient_id` | 患者编号 |
| `linear_predictor` | 模型输出的线性预测值（对数几率尺度，即 logit(预测概率)） |
| `true_outcome` | 真实结局（1/0） |
| `dataset_source` | 标注该记录来自开发集还是某个验证队列 |

有了这份表后，可以直接从开发集的 `linear_predictor` 列估计经验分布的 $\mu, \sigma$（或改用核密度估计代替正态假设），再代入 `estimation_ss_2` 的模拟流程。

脚本主体的标量参数：

| 参数名 | 含义 |
|---|---|
| `MU`, `SIGMA` | 假设的验证人群 $LP$ 分布参数 |
| `gamma`, `S` | 假设的真实校准模型参数（`gamma=0, S=1` 为良好校准；`S=0.9` 为轻度失校准情形） |
| `n_sims` | 模拟重复次数 |
| `nb_thresholds` | 净获益关注的阈值列表 |

---

## 七、`estimation_ss_1`：外部验证样本量（闭式解，二分类结局）

### 7.1 适用阶段
与 `estimation_ss_2` 目标相同（评估外部验证精度是否足够），但采用**闭式公式**而非模拟，计算速度更快，也是 `pmvalsampsize` 官方软件包所采用的方法。二者理论上应给出接近的结论，可互相印证。

### 7.2 数学原理逐步展开

**符号定义：**

| 符号 | 含义 |
|---|---|
| $N$ | 验证集样本量 |
| $\varphi$ | 验证人群中真实/预期的结局比例 |
| $C$ | 预期真实 c 统计量（区分度） |
| $O/E$ | 观察结局数 / 预期（模型预测）结局数之比，理想值1 |
| $\beta$（校准斜率，本节用 $\lambda$ 避免与检验效能符号冲突） | 理想值1 |
| $p_t$ | 临床决策相关的概率阈值，用于净获益计算 |

**准则1：O/E 比值精度**

$$N = \frac{1-\varphi}{\varphi \cdot \left[SE(\ln(O/E))\right]^2}$$

反查（已知 $N$，求达成的 $SE(\ln(O/E))$，再换算为 O/E 尺度上的置信区间）：

$$SE(\ln(O/E))_{achieved} = \sqrt{\frac{1-\varphi}{N\varphi}}$$

$$\text{95\% CI}_{O/E} = \exp\left[\pm 1.96 \times SE(\ln(O/E))_{achieved}\right] \times (O/E)_{assumed}$$

**准则2：校准斜率精度**（涉及 Fisher 信息矩阵）

$$N = \frac{I_\alpha}{SE(\lambda)^2 \left(I_\alpha I_\lambda - I^2_{\alpha,\lambda}\right)}$$

其中 $I_\alpha, I_{\alpha,\lambda}, I_\lambda$ 分别是以下三个量在验证人群线性预测值（$LP_i$）分布上的期望：

$$a_i = \frac{\exp(\alpha + \lambda \cdot LP_i)}{\left[1+\exp(\alpha + \lambda \cdot LP_i)\right]^2}, \quad
b_i = \frac{LP_i \cdot \exp(\alpha + \lambda \cdot LP_i)}{\left[1+\exp(\alpha + \lambda \cdot LP_i)\right]^2}, \quad
c_i = \frac{(LP_i)^2 \cdot \exp(\alpha + \lambda \cdot LP_i)}{\left[1+\exp(\alpha + \lambda \cdot LP_i)\right]^2}$$

假设模型良好校准时 $\alpha=0, \lambda=1$。

**准则3：c 统计量精度（Newcombe 公式，闭式解）**

$$SE(C) = \sqrt{\frac{C(1-C)\left[1 + \left(\dfrac{N}{2}-1\right)\dfrac{1-C}{2-C} + \left(\dfrac{N}{2}-1\right)\dfrac{C}{1+C}\right]}{N^2 \varphi(1-\varphi)}}$$

这是**闭式解**（不需要迭代），直接代入 $N, C, \varphi$ 即可求 $SE(C)$；反过来求 $N$ 则需要数值迭代（脚本内部处理）。

**准则4：标准化净获益（standardized net benefit, sNB）精度**（Marsh et al. 公式）

$$N = \frac{1}{SE(sNB_{p_t})^2}\left[\frac{sens(1-sens)}{\varphi} + w^2\frac{spec(1-spec)}{1-\varphi} + \frac{w^2(1-spec)^2}{\varphi(1-\varphi)}\right]$$

其中：

$$w = \frac{1-\varphi}{\varphi} \times \frac{p_t}{1-p_t}$$

**最终取四准则中所需 $N$ 的最大值。**

### 7.3 需要的数据/输入

同样是纯参数化输入，**不强制要求 CSV**，但如果你希望脚本从真实预测结果自动估计 $\varphi$、$C$，建议准备如下格式的 CSV（每行一位患者）：

| 列名 | 含义 | 示例值 |
|---|---|---|
| `patient_id` | 患者编号 | P001 |
| `predicted_prob` | 模型输出的预测概率 $\hat{p}_i$ | 0.42 |
| `true_outcome` | 真实结局（1=pCR，0=非pCR） | 1 |
| `cohort` | 所属队列标签 | Cohort_3 |

脚本主体使用的标量参数：

| 参数名 | 含义 |
|---|---|
| `n`（Cohort 2/3/4的N） | 反查时代入的实际样本量 |
| `phi` | 验证队列结局比例 |
| `c_stat` | 假设的真实c统计量 |
| `beta_params` | 假设的预测概率 Beta 分布参数 $(a, b)$，用于生成/近似线性预测值分布 |
| `thresholds` | 关注的临床决策阈值列表（用于净获益计算） |

---

## 八、从 c 统计量反推 $R^2_{cs}$（连接 ss_3 与 ss_2/ss_1 的桥梁）

如果你已有开发/内部验证阶段报告的 c 统计量，而 `estimation_ss_3` 需要的输入是 $R^2_{cs}$，可以用 Riley et al. 提供的近似换算关系（基于假设线性预测值服从正态分布的推导）：

$$\delta = \sqrt{2} \times \Phi^{-1}(C)$$

$$R^2_{cs} \approx 1 - \exp\left(-\frac{\delta^2}{2}\right) \times \text{（视具体换算公式版本，建议使用 R 中 `pmsampsize` 包内置换算函数核实）}$$

**注意**：这一换算本身依赖正态假设，属于近似值，建议只作为无更优信息时的保守起点，若有更精确的既往报告 $R^2_{cs}$ 应优先使用。

---

## 九、`estimation_ss_4`：模型间头对头比较样本量（两个相关AUC比较）

### 9.1 适用阶段
当研究问题从"这一个模型准不准"变成"我的模型 A 是否显著优于模型 B（在同一批患者身上比较）"时使用。这是 DeLong 检验（1988）的样本量对应版本。

### 9.2 数学原理逐步展开

**符号定义：**

| 符号 | 含义 |
|---|---|
| $X_k$ | 病例组（如 pCR 阳性）中生物标志物/模型 $k$（$k=1,2$）的取值 |
| $Y_k$ | 对照组（如 pCR 阴性）中生物标志物/模型 $k$ 的取值 |
| $\theta_k = P(X_k > Y_k)$ | 模型 $k$ 的真实 AUC |
| $\delta_k$ | 对数变换+标准化后，模型 $k$ 的标准化效应量，$X_k \sim N(\delta_k, 1)$，$Y_k \sim N(0,1)$ |
| $\rho$ | 两个模型预测分数之间的相关系数（假设病例组和对照组内相关系数相同） |
| $\gamma$ | 病例（结局阳性）在总体中的比例 |
| $m, n$ | 分别为病例组、对照组样本量，$m+n=N$，$m/N \approx \gamma$ |

**步骤1：AUC 与标准化效应量 $\delta_k$ 的一一对应关系**

$$\theta_k = \Phi\left(\frac{\delta_k}{\sqrt{2}}\right) \quad \Longleftrightarrow \quad \delta_k = \sqrt{2} \times \Phi^{-1}(\theta_k)$$

（这是标准的正态-正态模型下 AUC 公式，与论文 Table 1 中的对照表一致，如 $\delta=1.0 \to \theta=0.760$）

**步骤2：计算 $\sigma_k^2(\varepsilon)$**（单个模型的方差分量）

$$\sigma_k^2(\varepsilon) = \int_{-\infty}^{\infty} \phi(x-\delta_k)\Phi^2(x)\,dx - \theta_k^2$$

（脚本内用 `scipy.integrate.quad` 数值积分求解）

**步骤3：计算 $\sigma_{12}(\varepsilon)$**（两模型之间的协方差分量）

$$\sigma_{12}(\varepsilon) = \iint \Phi(x_1)\Phi(x_2)\,\phi_\rho(x_1-\delta_1, x_2-\delta_2)\,dx_1\,dx_2 - \theta_1\theta_2$$

其中 $\phi_\rho(\cdot,\cdot)$ 是均值为0、方差为1、相关系数为 $\rho$ 的二元正态密度函数。（脚本内用 `scipy.integrate.dblquad` 二维数值积分求解）

**步骤4：计算总方差 $v$**

$$v = \frac{1}{\gamma(1-\gamma)}\left[\sigma_1^2(\varepsilon) + \sigma_2^2(\varepsilon) - 2\sigma_{12}(\varepsilon)\right]$$

**步骤5：前瞻求解所需样本量**

设检验目标为双侧 $\alpha$ 水平、检验效能 $1-\beta$，检测的 AUC 差值为 $\Delta = \theta_1 - \theta_2$：

$$N = \frac{v\left(z_{1-\alpha/2} + z_{1-\beta}\right)^2}{\Delta^2}$$

$$m = \lceil \gamma N \rceil, \quad n = N - m$$

**步骤6：反查已知 $N$ 时的检验效能**

$$1-\beta_{achieved} = \Phi\left(z_{1-\alpha/2}^{-1}... \right)$$

更准确地说，由前瞻公式反解：

$$1-\beta_{achieved} = \Phi\left(\frac{\sqrt{N}\,|\Delta|}{\sqrt{v}} - z_{1-\alpha/2}\right)$$

**步骤7：反查已知 $N$ 时能检测出的最小 AUC 差值**（数值求根）

固定参考 AUC $\theta_1$，寻找满足下式的最小 $\theta_2$（脚本用 `scipy.optimize.brentq` 求根）：

$$\frac{v(\theta_1,\theta_2,\rho,\gamma)\left(z_{1-\alpha/2}+z_{1-\beta,target}\right)^2}{(\theta_2-\theta_1)^2} = N$$

### 9.3 需要的数据/输入

若你想从真实数据（而非纯假设值）估计 $\rho$（两模型预测分数的相关系数），建议准备如下 CSV：

| 列名 | 含义 |
|---|---|
| `patient_id` | 患者编号（必须保证模型A、模型B在同一患者上都有预测值——这是配对/相关设计的前提） |
| `model_A_score` | 模型A（如基线/竞品模型）的预测分数或概率 |
| `model_B_score` | 模型B（如 BC-BioMIXER）的预测分数或概率 |
| `true_outcome` | 真实结局（1/0） |

从这份表可以直接分别在病例组（`true_outcome=1`）和对照组（`true_outcome=0`）内计算 `model_A_score` 与 `model_B_score` 的 Pearson 相关系数，取两者的平均（或分别代入更精细的模型）作为 $\rho$ 的估计值，同时也能直接算出两模型各自的经验 AUC 作为 $\theta_1, \theta_2$ 的估计。

脚本主体的标量参数：

| 参数名 | 含义 |
|---|---|
| `THETA1`, `THETA2` | 两模型的（假设或经验）AUC |
| `RHO` | 两模型预测分数的相关系数 |
| `GAMMA` | 结局阳性比例 |
| `cohorts` | 待反查的实际验证队列样本量字典 |

---

## 十、Manuscript 中 Sample-Size Rationale 该怎么写

对应 TRIPOD+AI 条目10（"说明研究样本量是如何确定的，并论证该样本量足以回答研究问题，须包含任何样本量计算的细节"），建议按以下结构撰写，分别覆盖开发集与验证集：

### 10.1 开发集样本量段落模板

> The development cohort (Cohort 1, N = xxx) sample size was evaluated against the criteria proposed by Riley et al. for developing a clinical prediction model with a binary outcome [引用 estimation_ss_3 对应文献]. Assuming an anticipated outcome (pCR) proportion of xxx and xxx candidate predictor parameters, and a conservative anticipated Cox-Snell R² of xxx (derived from xxx), the minimum required sample size to achieve a shrinkage factor of ≥0.90, a mean absolute prediction error of ≤0.05, and an expected optimism in apparent R²Nagelkerke of ≤0.05 was xxx participants. [若样本量不足，补充：] The available development sample (N = xxx) fell short of this target by xxx participants, primarily constrained by the shrinkage criterion; this limitation is acknowledged in the Discussion (see Limitations).

### 10.2 外部验证集样本量段落模板

> External validation samples (Cohort xxx, N = xxx) were evaluated for their ability to precisely estimate model performance, following the approach of Riley et al. [引用 estimation_ss_1 对应文献] and, as a sensitivity check, the simulation-based approach of Snell et al. [引用 estimation_ss_2 对应文献]. Assuming an anticipated c statistic of xxx and an outcome proportion of xxx, the sample size required to achieve 95% confidence interval widths of ≤0.1 for the c statistic, ≤0.3 for the calibration slope, and ≤0.22 for the observed/expected ratio was xxx participants (xxx events), driven predominantly by the calibration slope criterion. The available validation cohort (N = xxx) achieved a calibration slope confidence interval width of approximately xxx, indicating [precise / limited] precision in this regard, which is discussed as a limitation where relevant.

### 10.3 模型比较段落模板（若涉及）

> To formally test whether [Model B] demonstrated superior discrimination compared with [Model A], a sample size calculation for comparing two correlated AUCs was conducted following Jung [引用 estimation_ss_4 对应文献], which is asymptotically equivalent to the DeLong test. Assuming AUCs of xxx and xxx for the two models respectively, a correlation of xxx between their predicted scores (estimated from xxx), and an outcome proportion of xxx, a sample size of xxx participants was required to detect this difference with 80% power at a two-sided alpha of 0.05. The available comparison cohort (N = xxx) achieved an estimated power of xxx for this comparison.

---

## 十一、四脚本一览表（速查）

| 脚本 | 阶段 | 发表年份 | 核心问题 | 关键输出 | 是否需要CSV |
|---|---|---|---|---|---|
| `estimation_ss_3` | 开发 | 2020 | 训练集够不够大，会不会过拟合 | B1-B4 四准则的达标情况，瓶颈项 | 否（纯参数） |
| `estimation_ss_2` | 外部验证（模拟法） | 2021 | 验证集精度够不够（c统计量/校准/净获益），允许显式模拟失校准情形 | 各指标的模拟平均95% CI宽度 | 否（纯参数），可选真实LP分布CSV |
| `estimation_ss_1` | 外部验证（闭式解） | 2023 | 同上，与 ss_2 互为印证/备选 | 四准则各自达成的95% CI宽度 | 否（纯参数） |
| `estimation_ss_4` | 模型比较 | 2024 | 两模型AUC差异能否被可靠检出 | 所需N / 反查效能 / 最小可检出差值 | 否（纯参数），可选配对预测值CSV |

---

*本教程配合脚本文件 `estimation_ss_1.py`、`estimation_ss_2.py`、`estimation_ss_3.py`、`estimation_ss_4.py` 使用，所有脚本内的占位参数（如 `PHI`、`THETA1`、`P` 等）均需替换为你的真实数据后方可用于正式的 manuscript 撰写。*
