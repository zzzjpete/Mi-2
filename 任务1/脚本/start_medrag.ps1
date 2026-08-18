# start_medrag.ps1 -- one-click entry into the medrag working environment
# Usage (NOTE the leading dot -- it activates the env in your CURRENT shell):
#     . E:\rag\start_medrag.ps1

# Make Chinese print correctly in this console
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

# 1) Ensure the Ollama server is running (start it on the E: model dir if not)
$ollamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
$ollamaDir = Split-Path $ollamaExe
if ($env:Path -notlike "*$ollamaDir*") { $env:Path = "$ollamaDir;$env:Path" }
if (-not (Get-Process -Name 'ollama' -ErrorAction SilentlyContinue)) {
    Write-Host "Starting Ollama server..." -ForegroundColor Yellow
    $env:OLLAMA_MODELS = "E:\rag\ollama\models"
    Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden
    Start-Sleep -Seconds 2
} else {
    Write-Host "Ollama server already running." -ForegroundColor DarkGray
}

# 2) Activate the conda env 'medrag'
& "C:\Users\Jinxi Zhang\miniconda3\shell\condabin\conda-hook.ps1"
conda activate medrag

# 3) Show available commands
Write-Host ""
Write-Host "medrag ready (Python 3.11, GPU enabled)" -ForegroundColor Green
Write-Host "------------------------------------------------------------"
Write-Host "Chat with model (simplest):  ollama run qwen3:8b"
Write-Host "Verify LLM inference:        python E:\rag\scripts\验证本地模型.py"
Write-Host "Verify data parsing:         python E:\rag\scripts\解析数据管线.py"
Write-Host "List installed models:       ollama list"
Write-Host "Show GPU usage:              ollama ps"
Write-Host "------------------------------------------------------------"
