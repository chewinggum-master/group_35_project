# FIPO 在 GSM8K 上的分析與討論

## 實驗設定總結

### 模型配置
- **Generator**: meta-llama/Llama-2-7b-hf (8-bit quantization)
- **Optimizer**: allenai/tulu-2-dpo-13b (4-bit quantization)
- **Dataset**: GSM8K test set (1,319 questions)

### 實驗方法對比

| 方法 | 描述 | 準確率 | 正確題數 |
|------|------|--------|----------|
| **Naive Baseline** | 無 FIPO，使用固定 prompt + 3-shot | 待測試 | 待測試 |
| **Task-level FIPO** | 單一全局優化指令 + 3-shot | **11.83%** | 156/1319 |
| **Instance-level FIPO** | 每題獨立優化 prompt | **1.82%** | 24/1319 |

---

## 主要發現

### 1. Task-level vs Instance-level 的差異

#### Task-level FIPO (單一優化指令)
- **流程**:
  1. 使用 meta-prompt 請 optimizer 生成一個全局數學推理指令
  2. 該指令儲存於 `data/fipo_instruction_13b.txt`
  3. 在所有測試題上使用同一指令 + 3-shot few-shot examples
  4. Generator greedy decoding 生成答案

- **優勢**:
  - 計算成本低（只需優化一次）
  - 指令穩定、可重複使用
  - 與 few-shot learning 結合良好
  - 達到 **11.83% 準確率**

- **限制**:
  - 指令需人工設計 seed prompt
  - 無法針對個別題目特性調整
  - 不完全符合原論文 instance-level 設定

#### Instance-level FIPO (逐題優化)
- **流程**:
  1. 對每個測試題生成專屬 Silver Prompt (question + exemplar)
  2. Optimizer 針對該題生成 Golden Prompt
  3. Generator 使用 Golden Prompt 回答該題

- **挑戰**:
  - Optimizer 輸出品質不穩定
  - 常見問題:
    - 產生 "cheating prompts"（直接給答案）
    - 輸出格式不一致（包含多餘文字、格式標記）
    - 過度複雜或錯誤的推理指導
  - 計算成本高（1,319 題需 1,319 次優化）
  - 當前僅達到 **1.82% 準確率**

---

## 深入分析

### 2. Instance-level 表現不佳的原因

#### (a) Optimizer 行為問題
從實驗 logs 觀察到的典型問題:

**問題類型 1: Cheating Prompts**
```
原題: "Janet's ducks lay 16 eggs per day..."
優化後 prompt: "...Step 4: Write '#### ' followed by the final numeric answer on the last line."
Generator 輸出: "#### $18" (直接給答案，無推理過程)
```

**問題類型 2: 過度指導 / 錯誤指導**
```
原題: "A robe takes 2 bolts of blue fiber and half that much white fiber..."
優化後 prompt: "...calculate half the amount of white fiber and add it to the 2 bolts of blue fiber..."
Generator 輸出: 錯誤計算（混淆加法與減法）
```

**問題類型 3: 格式污染**
- Optimizer 經常輸出包含 "Golden Prompt:", "Answer:", markdown 標記等
- 雖已添加清理機制，但仍有殘留影響 generator 理解

#### (b) 模型容量限制
- Llama-2-7B 在數學推理上本身能力有限（GSM8K SOTA ~50-60% with larger models）
- 即使有完美 prompt，7B 模型可能無法執行複雜多步推理
- 8-bit quantization 可能進一步降低推理能力

#### (c) Template 設計問題
- 雖已修正 placeholder (S_P, O_C, G_N)，但 meta-prompt 的設計可能不夠引導 optimizer 生成有效指令
- Optimizer 傾向生成過於籠統或過於具體的指令

---

### 3. Task-level 成功的原因

#### (a) Few-shot 的穩定性
- 3-shot examples 提供穩定的推理模式
- 減少對單一 prompt 品質的依賴

#### (b) 人工種子指令
- Task-level 的全局指令由人工設計 seed prompt 引導
- 雖非完全自動化，但品質可控

#### (c) 成本效益
- 只需一次優化，降低累積錯誤風險
- 適合實際應用場景

---

## 與原論文比較

### 論文預期 vs 實際結果

| 指標 | 原論文預期 | 本實驗結果 | 差異分析 |
|------|------------|------------|----------|
| Instance-level效果 | 顯著提升 | 1.82%（極低） | 可能原因: (1) 模型容量不足 (2) meta-prompt設計未對齊 (3) 缺少discriminator評估機制 |
| Optimizer品質 | 穩定、有效 | 不穩定、格式問題 | 可能原因: 13B模型能力限制，需更大模型或更好prompt engineering |
| Generator表現 | 準確改善 | 未見改善 | 7B模型推理能力瓶頸 |

### 可能的差異來源

1. **模型選擇差異**
   - 原論文可能使用更大模型 (如 70B+)
   - 本實驗受硬體限制使用 7B + quantization

2. **Discriminator 缺失**
   - 原論文 FIPO 框架包含 discriminator 評估優化效果
   - 本實驗未實作 discriminator，無品質過濾機制

3. **Meta-prompt 設計**
   - 原論文 meta-prompt 可能經過精心調校
   - 本實驗 meta-prompt 為自行設計，可能未對齊最佳實踐

4. **Optimizer 訓練**
   - 原論文可能對 optimizer 進行特定 fine-tuning
   - 本實驗使用現成 Tulu-2-DPO-13B

---

## 研究方法的貢獻與限制

### 貢獻
1. **明確區分兩種 FIPO 應用範式**:
   - Task-level: 適合需要穩定、可重複使用指令的場景
   - Instance-level: 理論上更靈活，但實作挑戰大

2. **實證分析**:
   - 證實在資源受限環境下，task-level 更實用
   - 揭示 instance-level 的實作難點（optimizer 品質、計算成本）

3. **完整實驗日誌**:
   - 詳細記錄 per-question 結果、優化 prompts
   - 便於後續分析與改進

### 限制
1. **硬體限制**:
   - 無法測試更大模型（如 70B）
   - Quantization 可能影響結果

2. **未實作 discriminator**:
   - 缺少品質評估與過濾機制
   - 無法進行 prompt 迭代優化

3. **Meta-prompt 設計空間**:
   - 當前 meta-prompt 可能未充分探索
   - 需要更多 prompt engineering 嘗試

4. **Naive baseline 未完成**:
   - 需補充純粹無 FIPO 的 baseline 結果
   - 才能完整評估 FIPO 的實際效益

---

## 實務建議

### 對於實際應用者

1. **優先考慮 Task-level FIPO**:
   - 成本低、效果相對穩定
   - 適合需要大量推理任務的場景

2. **Instance-level 需要**:
   - 更大 optimizer 模型（建議 ≥ 30B）
   - Discriminator 機制過濾低品質 prompts
   - 充足計算資源

3. **Few-shot 的重要性**:
   - 即使有 FIPO，few-shot examples 仍是穩定推理的關鍵
   - 建議保留 3-5 個高品質 examples

### 未來改進方向

1. **Optimizer 改進**:
   - 嘗試更大模型（如 Llama-3-70B）
   - Fine-tune optimizer 專門生成數學推理 prompts
   - 添加輸出格式約束（如 JSON schema）

2. **Generator 提升**:
   - 測試更強數學推理模型（如 WizardMath, MAmmoTH）
   - 考慮 ensemble 多個 generators

3. **Pipeline 優化**:
   - 實作 discriminator 評估 prompt 品質
   - 添加 prompt 迭代優化機制
   - 探索 hybrid approach（結合 task-level 穩定性與 instance-level 靈活性）

4. **Prompt Engineering**:
   - 系統化測試不同 meta-prompt 設計
   - 添加更多約束與指引（避免 cheating prompts）

---

## 結論

本實驗通過實作兩種 FIPO 範式，揭示了理論與實踐的差距:

- **Task-level FIPO** 在資源受限環境下是實用選擇（11.83% vs naive baseline 待測試）
- **Instance-level FIPO** 雖概念上更符合原論文，但在 7B/13B 模型規模下面臨顯著挑戰（僅 1.82%）

核心問題在於 **optimizer 品質** 與 **generator 容量** 的雙重限制。要實現有效的 instance-level FIPO，需要:
1. 更大、更專業的 optimizer 模型
2. Discriminator 機制確保 prompt 品質
3. 更強的數學推理 generator

對於實務應用，建議:
- 小規模資源: Task-level FIPO + few-shot
- 大規模資源: Instance-level FIPO + discriminator + 大模型

未來工作將聚焦於補充 naive baseline、改進 meta-prompt 設計，以及探索 discriminator 機制的整合。

---

## 附錄: 實驗檔案

- **Task-level 腳本**: `fipo_gsm8k_13b_experiment.py`
- **Instance-level 腳本**: `fipo_gsm8k_13b_2stage.py`
- **Naive baseline**: `gsn8k_naive.py` (待運行)
- **結果檔案**:
  - Task-level: `runs/fipo_gsm8k_13b_3shot_1319.json`
  - Instance-level: `runs/gsm8k_instance_results_1319.json`
  - Logs: `runs/*.log`
