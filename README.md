# Mobile Price Classification with Ensemble Learning

結合多種分類演算法與整合學習堆疊方法的手機價格分類專題

## 📋 目錄
- [專題概述](#-專題概述)
- [資料集](#-資料集)
- [研究方法](#-研究方法)
- [使用的模型](#-使用的模型)
- [實驗結果](#-實驗結果)
- [安裝說明](#-安裝說明)
- [使用方法](#-使用方法)
- [重要發現](#-重要發現)
- [專題結構](#-專題結構)
- [結論](#-結論)

## 🎯 專題概述

本專題分析手機規格與價格之間的關係。使用 Kaggle 的 Mobile Price Classification 資料集，實作四種不同的分類演算法，並比較使用與不使用整合學習堆疊（Stacking）方法的效能表現。

研究目標是根據 20 種不同的手機規格，預測手機價格區間（低階、中階、高階、非常高階）。

## 📊 資料集

- **來源**: Kaggle Mobile Price Classification 資料集
- **特徵數量**: 21 個（20 個特徵 + 1 個標籤）
- **目標變數**: `price_range` (0: 低階, 1: 中階, 2: 高階, 3: 非常高階)
- **資料分割**: 
  - 第一階訓練集: 50%
  - 第二階訓練集: 25%
  - 驗證集: 25%

### 關鍵特徵

根據相關係數分析與隨機森林特徵重要性評估：

1. **RAM（記憶體）** - 最重要特徵
2. **Battery Power（電池容量）** - 次重要特徵
3. **Pixel Width（螢幕寬度像素）** - 第三重要
4. **Pixel Height（螢幕高度像素）** - 第四重要

## 🔬 研究方法

### 資料前處理

- 資料集無缺失值
- 使用 Pandas `corr()` 進行特徵相關性分析
- 使用隨機森林評估特徵重要性

### 模型訓練流程

1. 使用第一階訓練集訓練四種基礎模型
2. 在第二階訓練集上產生預測結果
3. 將預測結果作為特徵訓練次級模型（Stacking）
4. 在驗證集上評估所有模型效能

## 🤖 使用的模型

### 基礎分類器

#### 1. 支持向量機分類器（Support Vector Classifier, SVC）
- 在高維特徵空間中尋找最佳超平面
- 使用核函數處理非線性問題
- **優點**：高維度資料表現優異、記憶體效率高
- **缺點**：大型資料集計算成本高

#### 2. 隨機森林分類器（Random Forest Classifier, RFC）
- 基於多個決策樹的整合學習方法
- **優點**：高準確性、抗過擬合、可評估特徵重要性
- **缺點**：模型解釋性較低

#### 3. 決策樹分類器（Decision Tree Classifier, DTC）
- 樹狀結構的分類模型
- **優點**：直觀易理解、可處理數值與類別特徵
- **缺點**：容易過擬合、對資料變動敏感

#### 4. K 近鄰分類器（K Neighbors Classifier, KNN）
- 基於鄰近樣本的距離計算
- **優點**：簡單直觀、適用於多類別分類
- **缺點**：大型資料集計算耗時、易受離群值影響

### 整合學習方法

#### 堆疊（Stacking）

- 將多個初級模型的預測結果作為新特徵
- 使用次級模型進行最終預測
- **優點**：提升預測效能、捕捉複雜關係
- **缺點**：計算複雜度高、需要仔細調整參數

## 📈 實驗結果

### 評估指標

- **準確率（Accuracy）**: 整體預測正確率
- **精確率（Precision）**: 預測為正例中實際為正例的比例
- **召回率（Recall）**: 實際正例中被正確預測的比例
- **F1 分數（F1 Score）**: 精確率與召回率的調和平均值

### 基礎模型效能

| 模型 | 準確率 | 精確率 | 召回率 | F1 分數 |
|------|--------|--------|--------|---------|
| SVC  | 0.972  | 0.974  | 0.972  | 0.973   |
| RFC  | 0.880  | 0.883  | 0.883  | 0.883   |
| DTC  | 0.812  | 0.817  | 0.815  | 0.816   |
| KNN  | 0.916  | 0.919  | 0.917  | 0.918   |

### 整合模型效能（Stacking）

| 次級模型 | 準確率 | 精確率 | 召回率 | F1 分數 |
|----------|--------|--------|--------|---------|
| Ensemble SVC | 0.972 | 0.974 | 0.972 | 0.973 |
| Ensemble RFC | 0.960 | 0.963 | 0.960 | 0.961 |
| Ensemble DTC | 0.962 | 0.965 | 0.962 | 0.963 |
| Ensemble KNN | 0.950 | 0.952 | 0.950 | 0.951 |

### 效能比較

- **最佳基礎模型**: SVC（準確率 97.2%）
- **最佳整合模型**: Ensemble SVC（準確率 97.2%）
- **最大提升**: DTC 從 81.2% 提升至 96.2%（+15%）

## 💻 安裝說明

### 環境需求
```bash
Python 3.9+
```

### 安裝相依套件
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

或使用 requirements.txt：
```bash
pip install -r requirements.txt
```

### requirements.txt 內容
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
seaborn>=0.11.0
matplotlib>=3.4.0
```

## 🚀 使用方法

### 1. 克隆專案
```bash
git clone https://github.com/your-username/mobile-price-classification.git
cd mobile-price-classification
```

### 2. 準備資料

將 `train.csv` 和 `test.csv` 放置於 `./data/` 目錄下

### 3. 執行基礎模型訓練

開啟並執行 `final_project.ipynb`：
```python
# 載入資料
import pandas as pd
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# 按照 notebook 中的步驟執行訓練
```

### 4. 執行整合學習

開啟並執行 `final_en.ipynb`：
```python
# 使用 Stacking 方法進行整合學習
# 詳細步驟請參考 notebook
```

## 🔍 重要發現

### 特徵重要性分析

1. **RAM** 與 **Battery Power** 是影響手機價格最重要的兩個因素
2. 螢幕解析度（px_width、px_height）也對價格有顯著影響
3. Pandas `corr()` 與 Random Forest 特徵重要性評估結果高度一致

### 特徵重要性前十名比較

| 排名 | corr() 方法 | Random Forest 方法 |
|------|-------------|-------------------|
| 1    | ram         | ram               |
| 2    | battery_power | battery_power   |
| 3    | px_width    | px_height         |
| 4    | px_height   | px_width          |
| 5    | int_memory   | mobile_wt         |
| 6    | sc_w   | talk_time          |
| 7    | pc   | int_memory          |
| 8    | three_g   | pc          |
| 9    | sc_h   | sc_w          |
| 10   | fc   | clock_speed          |

### 模型表現

1. **SVC** 在所有基礎模型中表現最佳，各項指標均達 97%+
2. **Stacking** 方法成功提升較弱模型的效能，所有模型準確率皆達 95%+
3. **Decision Tree** 透過 Stacking 獲得最大效能提升（+15%）
4. **DTC** 在基礎模型中表現最差（81.2%），但在整合模型中表現優於 KNN

### 視覺化分析

- 價格與 RAM 呈現明顯正相關
- 價格與電池容量呈現正相關
- 四個價格區間在資料集中分布均勻，無類別不平衡問題

## 📁 專題結構
```
mobile-price-classification/
│
├── data/
│   ├── train.csv              # 訓練資料
│   └── test.csv               # 測試資料
│
├── final_project.ipynb        # 基礎模型實作
├── final_en.ipynb             # 整合學習實作
├── s1080730_s1080741_HW4.pdf  # 完整報告
├── requirements.txt           # Python 套件需求
└── README.md                  # 專題說明文件
```

## 📝 結論

本研究成功運用四種分類演算法與整合學習方法進行手機價格分類：

### 主要成果

1. **特徵分析**：確認 RAM 與電池容量為最關鍵特徵，與市場實際情況相符
2. **模型比較**：SVC 表現最佳（97.2%），DTC 表現最弱（81.2%）
3. **整合提升**：Stacking 方法有效提升模型效能，特別是對原本較弱的模型
4. **實用價值**：所有整合模型準確率皆達 95%+，可實際應用於手機價格預測

### 研究貢獻

- 驗證了整合學習在手機價格分類問題上的有效性
- 提供了特徵重要性的多角度分析方法
- 展示了 Stacking 方法在提升弱分類器效能上的優勢

### 未來展望

- 嘗試更多整合學習方法（如 Boosting、Bagging）
- 加入深度學習模型進行比較
- 擴展至即時價格預測系統

## 👥 作者

**學生 ID**: s1080730, s1080741

## 📄 授權

本專題僅供學術研究使用

## 🙏 致謝

- 感謝 Kaggle 提供 Mobile Price Classification 資料集
- 感謝 scikit-learn 提供完整的機器學習工具

## 📧 聯絡方式

如有任何問題或建議，歡迎透過 GitHub Issues 與我們聯繫。

---

**最後更新**: 2024年12月
