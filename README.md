# AFG 守話 — 即時通話防詐

即時分析電話通話音訊、偵測詐騙手法並在通話中示警的工具。目前版本是 `antifraud_v3/`，
上一版 `antifraud_v2`（Streamlit、事後分析整通電話）已刪除，可從 git commit `0c66823` 找回。

## 為什麼重寫

`antifraud_v2` 的 `fraud_detection.py` 用「逐一指標加總門檻分數」的規則引擎判斷風險，這個設計
本身有結構性問題：正規化公式只用「有觸發」的指標權重當分母，導致單一微弱、常見的訊號（例如講話
語氣平靜）就能把某個詐騙模式的分數推到接近滿分，跟其他指標有沒有觸發無關。這正是「隨便講一句話
就被判定高風險」的根本原因——不是門檻數字設得不好，是這個公式的形狀本身就有問題，調數字沒有用。

`antifraud_v3` 用 LLM 推理（判別 → 反思 → 綜合三步驟）取代這個規則引擎，並把架構從「上傳整通電話
事後分析」改成「通話中即時分析、即時示警」。完整設計脈絡與決策理由見 `antifraud_v3/docs/DESIGN.md`。

## 架構

```
瀏覽器麥克風
  └─ AudioWorklet 擷取音訊 ──WebSocket──▶ FastAPI 後端
                                            │
                                     VAD 切句（silero-vad）
                                            │
                              ┌─────────────┼─────────────┐
                         語音辨識      聲學特徵分析      情緒辨識
                       (faster-whisper) (parselmouth)   (TIMNet)
                              └─────────────┼─────────────┘
                                            │
                          LLM 推理引擎：判別 → 反思 → 綜合
                             （Claude，可插拔 provider 介面）
                                            │
                              風險趨勢 + 示警 ──WebSocket──▶ 前端即時畫面
                                            │
                                  SQLite 通話紀錄／JSON 設定
```

- **不用固定秒數切句，用 VAD 偵測「一句話講完」**——這樣聲學特徵、情緒模型、LLM 推理才是對著完整
  語句分析，不是隨機切到一半。
- **偵測邏輯是 LLM 推理，不是規則引擎**——`reasoning/discriminate.py` → `reflect.py` →
  `synthesize.py`，反思步驟專門找「這個證據有沒有無辜的解釋」，這是舊版規則引擎完全沒有的機制。
- **示警要連續幾句風險都偏高才觸發**（debounce，可在設定畫面調），單一句話不會觸發一般示警；但一組
  「立即示警」關鍵字（要求驗證碼、要求轉帳等）可以跳過這個限制直接示警。
- **LLM 供應商可插拔**——`llm/base.py` 定義介面，`llm/claude_provider.py` 是目前唯一實作，換供應商
  只需要新增一個 class，不用動推理邏輯本身。
- **情緒辨識（TIMNet）刻意當弱訊號用**，不是硬性門檻——訓練語料是英/德/義等語言的表演式情緒語料，
  不是中文電話對話，實測對這個場景的預測不太準，所以只當 LLM 推理時的參考證據之一。
- **除了即時監聽，也可以上傳事後錄好的通話錄音分析**（`POST /api/upload-call`，前端「上傳錄音分析」
  分頁）——跑一樣的 VAD 切句 + per-chunk pipeline，差別只在於同步跑完整段錄音、一次回傳結果，而不是
  像即時通話一樣用 WebSocket 逐句推送（批次分析不像即時監聽那樣對延遲敏感，見
  `antifraud_v3/server/upload.py` 開頭的說明）。結果一樣會存進歷史紀錄，用 `source` 欄位跟即時通話
  區分開來。

## 安裝與設定

```bash
conda create -n AFG python=3.10   # 或用你自己的 venv
conda activate AFG
pip install -r antifraud_v3/requirements.txt
cp antifraud_v3/.env.example antifraud_v3/.env
```

**LLM 認證**（擇一）：
- 有 Claude 訂閱、不想另外買 API 額度：安裝 [`ant` CLI](https://github.com/anthropics/anthropic-cli)，
  跑 `ant auth login`（互動式瀏覽器登入，跟你的帳號綁定）。`llm/claude_provider.py` 用空建構子
  `anthropic.Anthropic()`，會自動抓這組登入狀態，不用改設定。
- 有獨立 API 金鑰：填到 `antifraud_v3/.env` 的 `ANTHROPIC_API_KEY`。

## 執行

```bash
cd /home/tommy/Project/AFG
uvicorn antifraud_v3.server.main:app --reload
```

開瀏覽器連 `http://localhost:8000`，允許麥克風權限後按「開始監聽」。

**在手機瀏覽器上用**（在真實通話中即時監聽的實際使用情境）：瀏覽器只有在 `https://` 或
`http://localhost` 底下才會允許存取麥克風，用手機連到區網或公網上另一台機器的 `http://` 網址不會
跳出權限請求。跑 HTTPS 有兩種驗證過可行的做法（自簽憑證給區網用、或用 `cloudflared` 之類的通道服務
拿到一個公開的 HTTPS 網址），完整步驟見 [`antifraud_v3/docs/HTTPS.md`](antifraud_v3/docs/HTTPS.md)。

## 專案結構

```
antifraud_v3/
├── audio/          VAD 切句、聲學特徵擷取（parselmouth/librosa）、TIMNet 情緒推論、MFCC
├── asr/            faster-whisper 語音辨識
├── llm/            可插拔 LLM 供應商介面 + Claude 實作
├── reasoning/       判別/反思/綜合推理引擎、詐騙話術 rubric、Pydantic schema
├── pipeline/        單一 chunk 的處理流程整合、通話狀態機（risk trajectory、示警 debounce）
├── server/          FastAPI app、WebSocket 即時端點、REST API（歷史紀錄／設定／上傳分析）
├── storage/         SQLite 通話歷史、JSON 設定檔存取
├── frontend/        即時通話／上傳分析／歷史紀錄／設定四個畫面（純 HTML/CSS/JS，無框架）
├── models/          TIMNet 模型架構與訓練好的權重檔
├── eval/            沒有標註資料集時的自我測試框架（regression log + 真人語音測試集）
├── tests/           pytest 單元測試（見下方「測試」一節）
└── docs/            DESIGN.md（完整設計決策記錄）、HTTPS.md（手機瀏覽器連線設定）
```

## 測試

**自動化測試**（`antifraud_v3/tests/`）：

```bash
cd /home/tommy/Project/AFG
pytest                        # 完整套件，包含會打真的 Claude API 的 live_api 測試
pytest -m "not live_api"      # 只跑快速、不用網路的單元測試（CI/沒有 API 額度時用這個）
```

大部分測試用一個假的 `LLMProvider`（`tests/conftest.py` 的 `FakeLLMProvider`）跟暫存的
DB／設定檔路徑，不會碰到真正的 API 額度或 `antifraud_v3/data/` 底下的正式資料。少數標了
`@pytest.mark.live_api` 的測試會真的呼叫 Claude API，當作跟真實模型行為對齊的 smoke test。

**手動/端對端驗證**：已用真實中文語音（`edge-tts` 生成，`eval/test_clips/` 底下 scam/benign 各
10 段，涵蓋銀行/公務機關假冒、投資詐騙、假綁架等多種話術，以及冷靜語氣、慢語速、跟詐騙無關的抱怨
等已知容易誤判的正常對話情境）+ headless 瀏覽器（Playwright，模擬麥克風輸入／檔案上傳）跑過完整
流程，確認麥克風擷取、VAD 切句、語音辨識、聲學特徵、LLM 推理到示警全部串接正確。跑
`python -m antifraud_v3.eval.run_test_set` 會把每段測試音檔的判斷結果記錄到
`eval/regression_log.jsonl`，改動 prompt/rubric 後可以快速比對有沒有退步。

**已知還沒做的**：
- 語者分離——目前分不出通話中「誰在講話」，聲學/情緒特徵是整段混合訊號的統計量。
- AI 合成語音／換聲偵測——完全沒涵蓋這塊風險，見 `references/04-asvspoof5.md`。
- 歷史紀錄/設定畫面沒有帳號或多裝置同步機制，單機本地使用。
- 上傳分析目前只吃 `soundfile`（libsndfile）能直接解碼的格式（wav/flac/ogg）——手機錄音常見的
  m4a/mp3 需要先用 ffmpeg 轉檔，見 `server/upload.py` 的錯誤訊息。

## 文獻

`references/` 收錄了 10 篇支撐這次重寫設計決策的文獻（2023–2026，含 DOI/連結），涵蓋 TIMNet 原始
論文、跨語言情緒辨識落差、聲學壓力線索的科學證據強度、AI 換聲偵測、LLM 輔助詐騙偵測等主題，見
`references/README.md` 的索引與閱讀建議。
