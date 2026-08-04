# AFG 守話 — 即時通話防詐

即時分析電話通話音訊、偵測詐騙手法並在通話中示警的工具。目前版本是 `antifraud_v3/`，
上一版 `antifraud_v2`（Streamlit、事後分析整通電話）已刪除，可從 git commit `0c66823` 找回。

## 如何使用這個專案

1. **安裝環境**
   ```bash
   conda create -n AFG python=3.10   # 或用你自己的 venv
   conda activate AFG
   pip install -r antifraud_v3/requirements.txt
   ```
2. **設定 LLM**
   ```bash
   cp antifraud_v3/.env.example antifraud_v3/.env
   ```
   預設 `LLM_PROVIDER=claude_code`，不用填任何金鑰——只要這台機器的 `claude` CLI 是登入狀態
   （跑 `claude auth status` 確認 `authMethod: "claude.ai"`），就會走你的 Claude Pro/Max 訂閱額度，
   不是按 token 計費的 API。如果你有獨立的 API 金鑰想用，見下方「LLM 認證細節」。
3. **啟動伺服器**
   ```bash
   cd /home/tommy/Project/AFG
   uvicorn antifraud_v3.server.main:app --reload
   ```
4. **打開瀏覽器**：連到 `http://localhost:8000`，允許麥克風權限，按「開始監聽」即時分析通話；
   或用「上傳錄音分析」分頁分析事後錄好的錄音；「測試資料」分頁可以直接跑內建的範例音檔看效果。
5. **（選用）在手機瀏覽器上用**：瀏覽器只有在 `https://` 或 `http://localhost` 底下才會允許存取
   麥克風，用手機連到區網/公網上另一台機器的 `http://` 網址不會跳出權限請求，需要另外跑 HTTPS——
   完整步驟見 [`antifraud_v3/docs/HTTPS.md`](antifraud_v3/docs/HTTPS.md)。

**LLM 認證細節**：如果你有獨立的、有額度的 API 金鑰，可以切回原本直接呼叫 API 的路徑：把
`antifraud_v3/.env` 的 `LLM_PROVIDER` 改成 `claude`，並填 `ANTHROPIC_API_KEY`（或安裝
[`ant` CLI](https://github.com/anthropics/anthropic-cli) 跑 `ant auth login`）。這兩個設定
在前端「設定」畫面也可以直接切換，不用改檔案重開伺服器。

## 這個專案怎麼分析一通電話

一句話講完（VAD 偵測到停頓）就立刻分析聲學/語音/情緒，即時顯示；LLM 判斷則是整通電話結束後才跑一次：

- **VAD 切句**（silero-vad）——偵測「一句話講完」，不是固定秒數硬切。
- **語音辨識 ASR**（faster-whisper）——即時把每句話轉成逐字稿。
- **聲學特徵分析**（parselmouth／librosa）——音高、音量、jitter/shimmer/HNR（聲音顫抖度）、
  語速與停頓變化，即時算出並畫成圖。
- **情緒辨識**（TIMNet）——即時標出每句話的情緒（生氣／恐懼／中性等），當作輔助證據，不是硬性門檻。
- **關鍵字立即示警**（純字串比對，不用 LLM）——通話進行中命中「要求驗證碼」「要求轉帳」等關鍵字
  立刻示警，這是通話中唯一即時的示警來源。
- **LLM 推理**（判別 → 反思 → 綜合三步驟，Claude）——通話結束後對整段逐字稿+聲學+情緒摘要跑一次，
  產出最終風險等級與說明；「反思」步驟專門找「這個證據有沒有無辜的解釋」，避免把正常對話誤判成詐騙。

完整設計脈絡與為什麼這樣分工（即時 vs. 事後一次）見下方「架構」與 `antifraud_v3/docs/DESIGN.md`。

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
                         語音辨識      聲學特徵分析      情緒辨識        ← 每句話即時跑，不用 LLM
                       (faster-whisper) (parselmouth)   (TIMNet)          即時推送到前端逐字稿
                              └─────────────┼─────────────┘
                                            │ （同時：關鍵字比對，命中立即示警，不用 LLM）
                                            │
                              …（通話進行中持續累積逐字稿／聲學／情緒）…
                                            │
                                      「結束監聽」
                                            │
                          LLM 推理引擎：判別 → 反思 → 綜合          ← 整通電話只跑這一次
                        （claude_code：走 Claude Pro/Max 訂閱額度，
                          或 claude：anthropic API，可插拔 provider 介面）
                                            │
                              最終研判 + 示警 ──WebSocket──▶ 前端即時畫面
                                            │
                                  SQLite 通話紀錄／JSON 設定
```

- **不用固定秒數切句，用 VAD 偵測「一句話講完」**——這樣聲學特徵、情緒模型、LLM 推理才是對著完整
  語句分析，不是隨機切到一半。
- **聲學／語音辨識／情緒分析是即時的，LLM 推理只在通話結束後跑一次**——原本每句話都跑一次
  `discriminate → reflect → synthesize`（判別 → 反思 → 綜合），對沒有付費 API 額度來說成本太高；
  現在通話中即時串流的只有 ASR/聲學/情緒（都是本地模型，不用 LLM），完整的 LLM 研判在「結束監聽」
  後對整通逐字稿跑一次，見 `antifraud_v3/pipeline/chunk_worker.py` 的 `run_final_analysis`。
- **偵測邏輯是 LLM 推理，不是規則引擎**——`reasoning/discriminate.py` → `reflect.py` →
  `synthesize.py`，反思步驟專門找「這個證據有沒有無辜的解釋」，這是舊版規則引擎完全沒有的機制。
- **「立即示警」關鍵字比對是通話進行中即時跑的**（要求驗證碼、要求轉帳等，純字串比對、不用 LLM，見
  `pipeline/call_state.py` 的 `check_live_hard_trigger`）——這是通話還在進行中時唯一即時的示警來源，
  因為完整的 LLM 研判被延後到通話結束才跑。
- **LLM 供應商可插拔**——`llm/base.py` 定義介面。預設是 `llm/claude_code_provider.py`（呼叫
  `claude` CLI，走 Claude Pro/Max 訂閱額度，不用按 token 付費的 API 金鑰）；`llm/claude_provider.py`
  是原本直接呼叫 anthropic API 的實作，有獨立 API 金鑰的話還是可以用。換供應商只需要新增一個
  class，不用動推理邏輯本身。
- **情緒辨識（TIMNet）刻意當弱訊號用**，不是硬性門檻——訓練語料是英/德/義等語言的表演式情緒語料，
  不是中文電話對話，實測對這個場景的預測不太準，所以只當 LLM 推理時的參考證據之一。
- **除了即時監聽，也可以上傳事後錄好的通話錄音分析**（`POST /api/upload-call`，前端「上傳錄音分析」
  分頁）——跑一樣的 VAD 切句 + 逐句訊號擷取 + 一次最終分析，差別只在於同步跑完整段錄音、一次回傳結果，
  而不是像即時通話一樣用 WebSocket 逐句推送（批次分析不像即時監聽那樣對延遲敏感，見
  `antifraud_v3/server/upload.py` 開頭的說明）。結果一樣會存進歷史紀錄，用 `source` 欄位跟即時通話
  區分開來。

## 專案結構

```
antifraud_v3/
├── audio/          VAD 切句、聲學特徵擷取（parselmouth/librosa）、TIMNet 情緒推論、MFCC
├── asr/            faster-whisper 語音辨識
├── llm/            可插拔 LLM 供應商介面 + Claude 實作
├── reasoning/       判別/反思/綜合推理引擎、詐騙話術 rubric、Pydantic schema
├── pipeline/        逐句訊號擷取 + 整通電話一次性最終分析、通話狀態機（逐字稿、示警狀態）
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
