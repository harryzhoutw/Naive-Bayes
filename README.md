# RFID 異常檢測

## 問題描述

有時候會收到未知的異常卡號資料（如 "0", "1", "3" 等），需要能判斷出來且不能使用 hardcode，不然出現未知的資料時又會判斷不到。

**挑戰**： 無法預知異常卡號

## 解決方案

使用**樸素貝葉斯 (Naive Bayes)** 的**高斯樸素貝氏 (Gaussian Naive Bayes)** 演算法來進行異常檢測。

### 演算法核心：高斯樸素貝氏 (Gaussian Naive Bayes)

高斯樸素貝氏是一種分類演算法，它假設每個特徵都服從高斯（常態）分佈，並且特徵之間相互獨立。在我們的異常檢測情境中，它通過以下步驟來學習「正常」RFID 卡號的模式：

1.  **特徵提取與量化**: 對於每個輸入的 RFID 卡號，我們提取多個數值特徵（例如長度、熵等）。這些特徵將作為高斯樸素貝氏模型的輸入。
2.  **學習正常分佈的參數**: 模型會分析所有「正常」訓練資料集的每個特徵。對於每一個特徵，它會計算其在訓練集中的**平均值 (Mean)** 和**標準差 (Standard Deviation)**。這些統計量共同定義了「正常」RFID 卡號各特徵的高斯分佈。
    *   例如，如果「長度」這個特徵在正常卡號中平均為 8 且標準差為 1，那麼一個長度為 8 的卡號在該特徵上會被認為非常「正常」，而一個長度為 2 的卡號則會被認為非常「不正常」。
3.  **計算對數似然 (Log-Likelihood) 並設定閾值**: 當一個新的 RFID 卡號進入時，模型會根據其提取的特徵和之前學到的正常分佈參數，計算該卡號屬於「正常」類別的對數似然分數。這個分數反映了該卡號與正常模式的匹配程度。分數越高，越可能是正常。模型會基於訓練資料的對數似然分數設定一個閾值，低於此閾值的卡號即被判定為異常。

## 特徵工程 (Feature Engineering)

我們定義了以下六個特徵來量化 RFID 卡號的特性：

| 特徵 | 說明 |
|------|------|
| 長度 (Length) | RFID 字串的總長度。 |
| 不重複字元數 (Unique Chars) | RFID 字串中不同字元的數量。 |
| 熵值 (Entropy) | 衡量 RFID 字串中字元分佈的混亂程度。高熵值表示字元分佈更均勻。 |
| 數值(Log-Value) | 將 RFID 字串視為十六進制數轉換為十進制，然後取自然對數。用於捕捉數值大小的特性。 |
| 字母比例 (Alpha Ratio) | RFID 字串中字母字元佔總長度的比例。 |
| 重複字元比例 (Repeat Ratio) | RFID 字串中最常見字元出現次數佔總長度的比例。 |

## 訓練與測試資料集

### 訓練資料集 (Training Dataset)

用於訓練模型，學習「正常」RFID 卡號的特徵分佈：

```json
normalRfids: [
    "A1B2C3D4", "E5F6A7B8", "C9D0E1F2", "F3A4B5C6", "D7E8F9A0", "B1C2D3E4", "A5B6C7D8", "E9F0A1B2", "C3D4E5F6", "F7A8B9C0",
    "A1B2C3D456", "E5F6A7B890", "C9D0E1F2AB", "F3A4B5C6CD", "1A2B3C4D", "5E6F7A8B", "9C0D1E2F", "3A4B5C6D", "7E8F9A0B",
    "2C3D4E5F", "6A7B8C9D", "0E1F2A3B", "4C5D6E7F", "8A9B0C1D", "A1B2C3D4E5", "F6A7B8C9D0", "E1F2A3B4C5", "D6E7F8A9B0"
]
```

### 測試資料集 (Testing Dataset)

#### 正常卡號檢查 (Normal RFID Check)

預期被判定為正常的 RFID 卡號：

```json
normalCheck: [
    "A1B2C3D4", "E5F6A7B8", "1A2B3C4D", "A1B2C3D456"
]
```

#### 異常卡號檢查 (Anomaly RFID Check)

預期被判定為異常的 RFID 卡號：

```json
anomalyCheck: [
    "0", "1", "000", "00000000", "11111111", "FFFFFFFF", "0000000000000000"
]
```

## 測試結果 (Test Results)

模型在判斷正常與異常卡號時，對數似然 (Log-Likelihood) 分數會呈現明顯差異。

| 類型 | RFID 範例 | 預期結果 | Log-Likelihood (範例值) |
|------|-----------|----------|--------------------------|
| 正常 | `A1B2C3D4` | 正常     | `-5.2` (接近 0, 較高)    |
| 異常 | `0`        | 異常     | `-350.12` (遠離 0, 較低) |
| 異常 | `1`        | 異常     | `-402.40`                |
| 異常 | `00000000` | 異常     | `-369.45`                |
| 異常 | `11111111` | 異常     | `-337.05`                |
| 異常 | `FFFFFFFF` | 異常     | `-336.27`                |

這些結果表明，正常 RFID 的對數似然分數顯著高於異常 RFID，這使得模型能夠有效地將兩者區分開來。

## 程式碼實作與解析

此部分將程式碼與核心的演算法邏輯結合說明。

### 1. 建立驗證器 (模型訓練)

當你用訓練資料去建立 `RfidGaussianNaiveBayesService` 物件時，模型的核心訓練邏輯就會在建構子 (`constructor`) 中自動執行：

1.  **特徵抽取**: 程式會遍歷所有訓練卡號，並針對每一個卡號計算出六個特徵的數值（長度、不同字元數、熵等）。
    ```cpp
    // 遍歷所有訓練資料，並呼叫 extractFeatures 抽取特徵
    std::vector<std::array<double, NUM_FEATURES>> allFeatures;
    for (const auto& rfid : uniqueList) {
        allFeatures.push_back(extractFeatures(rfid));
    }
    ```
2.  **學習分佈**: 接著，計算每一種特徵在所有訓練資料中的**平均值 (mean)** 與**標準差 (std)**。這些統計數據就是模型學到的「正常卡號特徵分佈」，結果會對應到「特徵工程」表格中的數據。
    ```cpp
    // 計算每個特徵的平均值與標準差
    for (size_t f = 0; f < NUM_FEATURES; ++f) {
        double sum = 0.0, sumSq = 0.0;
        for (const auto& features : allFeatures) {
            sum += features[f];
            sumSq += features[f] * features[f];
        }

        double mean = sum / allFeatures.size();
        double variance = (sumSq / allFeatures.size()) - (mean * mean);
        double std = std::max(std::sqrt(variance), 0.1); // 避免標準差為 0

        featureParams_[f] = {mean, std}; // 儲存模型的統計數據
    }
    ```
3.  **設定門檻**: 最後，程式會用學習到的分佈去計算每一筆訓練資料的 `Log-Likelihood`，並取其中最小的值再減去一個邊界值（margin），作為判斷異常的**門檻 (Threshold)**。
    ```cpp
    // 計算訓練資料中最小的 Log-Likelihood 作為基礎門檻
    double minLogLikelihood = std::numeric_limits<double>::max();
    for (const auto& rfid : uniqueList) {
        double ll = calculateLogLikelihood(extractFeatures(rfid));
        minLogLikelihood = std::min(minLogLikelihood, ll);
    }

    // 最小分數再減去一個邊界值，得到最終門檻
    threshold_ = minLogLikelihood - 1.0;
    ```

    ```cpp
    // 2. 建立驗證器物件，此步驟將自動觸發模型訓練
    RfidGaussianNaiveBayesService validator(normalRfids);
    ```

### 2. 驗證新卡號 (執行預測)

呼叫 `validate` 方法來檢測一個新的 RFID 卡號。這個函式會執行以下預測邏輯：

1.  **抽取特徵**: 計算輸入卡號的六個特徵值。
    ```cpp
    // 取得標準化後的 RFID 字串
    std::string normalized = toUpperCase(trim(rfid));
    // 抽取特徵值
    auto features = extractFeatures(normalized);
    ```
2.  **計算 Log-Likelihood**: 使用上一步訓練好的高斯分佈模型（平均值和標準差），計算這個新卡號的特徵所對應的 `Log-Likelihood` 分數。此分數代表了這個卡號有多麼「像」一個正常的卡號。
    ```cpp
    // 計算 Log-Likelihood 分數
    double logLikelihood = calculateLogLikelihood(features);
    ```
3.  **比較門檻**: 將計算出的 `Log-Likelihood` 與模型內部儲存的 `threshold` 進行比較。
    *   如果 `Log-Likelihood >= threshold`，則判斷為 `isValid = true` (正常)。
    *   如果 `Log-Likelihood < threshold`，則判斷為 `isValid = false` (異常)。

    ```cpp
    // 3. 傳入要驗證的卡號來進行預測
    auto result = validator.validate("FFFFFFFF");
    ```

### 3. 解讀驗證結果

`validate` 方法會回傳一個 `ValidationResult` 結構，其中包含：
*   `isValid` (bool): `true` 代表正常，`false` 代表異常。
*   `confidence` (double): 基於分數轉換的信賴度 (0.0 ~ 1.0)。
*   `reason` (string): 包含計算出的 `Log-Likelihood` 和門檻值，方便偵錯。

```cpp
// 4. 根據回傳結果進行後續處理
if (result.isValid) {
    std::cout << "有效的 RFID，信賴度: " << result.confidence << std::endl;
} else {
    std::cout << "檢測到異常！原因: " << result.reason << std::endl;
}
// 輸出範例: 檢測到異常！原因: log-likelihood=-336.27 < threshold=-6.63345
```

```cmd
[info] === RFID Anomaly Detector ===
[info] Loaded config from: test/test_data.json
[info] [Model Init] Learning Gaussian distribution from training data:
[info] Length: mean=8.571428571428571, std=0.9035079029052586
[info] Distinct Chars: mean=8.535714285714286, std=0.8652886742561147
[info] Entropy: mean=3.084836598539247, std=0.1387195773107373
[info] Numeric Value: mean=23.151617680989762, std=2.8395167813331716
[info] Letter Ratio: mean=0.5, std=0.1
[info] Repeat Ratio: mean=0.12142857142857144, std=0.1
[info] [Model Init] Min log-likelihood: -5.633446680624724, Threshold: -6.633446680624724

[info] === Testing Normal RFIDs ===
[info] RFID: A1B2C3D4 -> VALID (confidence: 0.7754471605977108)
[info] RFID: E5F6A7B8 -> VALID (confidence: 0.7773493813341649)
[info] RFID: 1A2B3C4D -> VALID (confidence: 0.7564988310026651)
[info] RFID: A1B2C3D456 -> VALID (confidence: 0.5590939621454715)

[info] === Testing Anomaly RFIDs ===
[info] RFID: 0 -> INVALID (log-likelihood=-404.36151089258516 < threshold=-6.633446680624724)
[info] RFID: 1 -> INVALID (log-likelihood=-402.40100795251533 < threshold=-6.633446680624724)
[info] RFID: 00000000 -> INVALID (log-likelihood=-369.44901089258576 < threshold=-6.633446680624724)
[info] RFID: 11111111 -> INVALID (log-likelihood=-337.0496540392308 < threshold=-6.633446680624724)
[info] RFID: FFFFFFFF -> INVALID (log-likelihood=-336.2687848749721 < threshold=-6.633446680624724)
[info] RFID: 000 -> INVALID (log-likelihood=-388.2615108925855 < threshold=-6.633446680624724)
[info] RFID: 0000000000000000 -> INVALID (log-likelihood=-29606.49067650135 < threshold=-6.633446680624724)

[info] === Summary ===
[info] Anomaly detection rate: 7/7 (100%)
```