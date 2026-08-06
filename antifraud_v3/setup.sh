#!/usr/bin/env bash
# One-time environment setup for antifraud_v3. Run once before first use:
#   bash antifraud_v3/setup.sh
#
# What this does:
#   1. Creates/activates a conda env (skipped if you're already in a conda env or venv)
#   2. Installs Python deps, working around fairseq==0.12.2's broken PyPI metadata
#   3. Copies .env.example -> .env if missing
#   4. Checks (does not install) system packages, a CUDA GPU, and the claude CLI login state
#
# Safe to re-run — every step is idempotent.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANTIFRAUD_DIR="$REPO_ROOT/antifraud_v3"
ENV_NAME="AFG"
PYTHON_VERSION="3.10"

step() { echo; echo "==> $1"; }
warn() { echo "    [警告] $1"; }
ok()   { echo "    [OK] $1"; }

cd "$REPO_ROOT"

step "1/5 Python 環境"
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" != "base" ]; then
  ok "已在 conda 環境「$CONDA_DEFAULT_ENV」中，沿用這個環境，不另外建立。"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
  ok "已在 venv「$VIRTUAL_ENV」中，沿用這個環境，不另外建立。"
elif command -v conda >/dev/null 2>&1; then
  if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    ok "conda 環境「$ENV_NAME」已存在，沿用。"
  else
    echo "    建立 conda 環境「$ENV_NAME」（python $PYTHON_VERSION）..."
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
  fi
  # `conda activate` needs conda's shell function, not just the binary on PATH.
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
  ok "已啟用 conda 環境「$ENV_NAME」。"
else
  warn "沒有偵測到 conda，也不在任何 venv 裡。請自行建立虛擬環境後再跑這個腳本，例如："
  warn "  python3 -m venv .venv && source .venv/bin/activate && bash antifraud_v3/setup.sh"
  exit 1
fi

step "2/5 系統套件檢查（不會自動安裝，只提醒）"
for bin in ffmpeg; do
  if command -v "$bin" >/dev/null 2>&1; then
    ok "$bin 已安裝。"
  else
    warn "$bin 沒裝——上傳分析要吃 m4a/mp3 等非 wav/flac/ogg 格式時需要，請手動：sudo apt-get install ffmpeg"
  fi
done
if ldconfig -p 2>/dev/null | grep -q libsndfile; then
  ok "libsndfile 已安裝。"
else
  warn "libsndfile 沒偵測到——soundfile 套件需要它，請手動：sudo apt-get install libsndfile1"
fi

step "3/5 安裝 Python 套件"
# fairseq==0.12.2's PyPI metadata uses an old `omegaconf<2.1` version-specifier syntax that
# pip>=24.1 refuses to parse — this is the model's own documented setup requirement (see
# requirements.txt's own comment on the fairseq line), not something fixable from this script's
# side beyond pinning pip down for the install. Restored to latest afterward — pip's own version
# doesn't affect already-installed packages at runtime, so this is safe to do every run.
ORIGINAL_PIP_VERSION="$(python -m pip --version | awk '{print $2}')"
python -m pip install -q "pip==24.0"
python -m pip install -q -r "$ANTIFRAUD_DIR/requirements.txt"
python -m pip install -q --upgrade pip
ok "Python 套件安裝完成（pip 版本 $ORIGINAL_PIP_VERSION -> 24.0 -> 最新，僅安裝過程暫時降版）。"

step "4/5 .env 設定檔"
if [ -f "$ANTIFRAUD_DIR/.env" ]; then
  ok ".env 已存在，不覆蓋。"
else
  cp "$ANTIFRAUD_DIR/.env.example" "$ANTIFRAUD_DIR/.env"
  ok "已從 .env.example 建立 .env（預設 LLM_PROVIDER=claude_code，走 claude CLI 訂閱額度）。"
fi

step "5/5 GPU 與 claude CLI 檢查（不會自動安裝，只提醒）"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  ok "偵測到 GPU：$GPU_INFO"
  echo "    Line 1（XLS-R-2B 合成語音偵測，fp16 常駐約 4.3GB）+ Line 2 選 qwen2audio 時（4-bit 常駐約"
  echo "    6.6GB）建議至少 12GB VRAM；Line 2 選 claude（預設）則不吃 GPU，見下方說明。"
else
  warn "沒偵測到 NVIDIA GPU——Line 1（AI 合成語音偵測）與本機 Qwen2Audio 版 Line 2 都需要 CUDA GPU"
  warn "才能在合理時間內跑完；沒有 GPU 的話，Settings 裡把 Line 2 留在預設的 claude（純文字判定，"
  warn "不吃 GPU）可用，但 Line 1 目前沒有 CPU 後備路徑。"
fi
if command -v claude >/dev/null 2>&1; then
  ok "claude CLI 已安裝在 PATH。請確認已登入：claude auth status（要看到 authMethod: \"claude.ai\"）。"
else
  warn "claude CLI 沒裝——預設的 LLM_PROVIDER=claude_code 跟新加的 Line2=claude 都需要它。"
  warn "安裝方式見 https://docs.claude.com/claude-code ，或改用 .env 的 LLM_PROVIDER=claude"
  warn "（直接呼叫 anthropic API，需要 ANTHROPIC_API_KEY，按 token 計費）。"
fi

echo
echo "==> 完成。啟動伺服器："
echo "    cd $REPO_ROOT && uvicorn antifraud_v3.server.main:app --reload"
echo "    然後打開 http://localhost:8000"
