# -*- coding: utf-8 -*-
# 迁移_传输到Mac.ps1 —— 把 E:\rag 里「不可重建」的部分推到 Mac 的 SMB 共享
#
# 设计原则：只搬不可重建的，能在 Mac 侧重建的一律不搬（conda 是 Windows 二进制、
# ollama 模型本地 pull 更快、pip-cache 与运行期 db 无价值）。
# 清单里每一项都写了「为什么必须搬」，改清单前先读那一列。
#
# 用法：
#   # 先看要搬什么、多大，不传任何东西
#   & .\scripts\迁移_传输到Mac.ps1 -Dest \\Jinxis-MacBook-Pro\rag -DryRun
#
#   # 真传（可随时 Ctrl-C，重跑会跳过已完成的文件、续传断掉的大文件）
#   & .\scripts\迁移_传输到Mac.ps1 -Dest \\Jinxis-MacBook-Pro\rag
#
#   # 连建库保险文件 merged_4m.parquet(15G) 一起搬（Mac 要能独立重建向量库时才需要）
#   & .\scripts\迁移_传输到Mac.ps1 -Dest \\Jinxis-MacBook-Pro\rag -IncludeParquet
#
# 目标也可以是已挂载的盘符（先在资源管理器把 Mac 共享映射成 M:），例如 -Dest M:\rag

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Dest,

    # data/vectors/merged_4m.parquet (15G)。不搬 = Mac 上永远无法重建 chroma 库与文献目录。
    [switch]$IncludeParquet,

    # 只统计与打印，不复制
    [switch]$DryRun,

    # 连原始语料一起搬（+118G）。除非 Windows 这台要退役，否则不需要。
    [switch]$IncludeCorpus
)

$ErrorActionPreference = 'Stop'
$Src = 'E:\rag'

if (-not (Test-Path $Src)) { throw "源目录不存在: $Src" }

$LogDir = Join-Path $Src 'logs\migrate'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$LogFile = Join-Path $LogDir ("传输_" + (Get-Date -Format 'yyyyMMdd_HHmmss') + ".log")

# ── 排除清单（根目录那一趟用） ────────────────────────────────────────────────
# 这些要么不可移植、要么 Mac 侧重建更快、要么纯属运行期垃圾
$RootExcludeDirs = @(
    "$Src\data"          # data 下逐项挑，见下面 $Manifest
    "$Src\conda"         # 5.63G Windows 二进制，Mac 上装不了，必须重建
    "$Src\ollama"        # 4.87G，Mac 上 `ollama pull qwen3:8b` 比网传快
    "$Src\pip-cache"     # 2.61G 纯缓存
    "$Src\logs"          # 运行期日志，含本脚本自己的日志
)
$RootExcludeFiles = @('~$*', '*.pyc')

# ── 搬运清单 ─────────────────────────────────────────────────────────────────
# Kind: 'tree' 整目录 / 'file' 单文件
$Manifest = New-Object System.Collections.ArrayList

[void]$Manifest.Add(@{
    Kind = 'tree'; Src = $Src; Rel = ''
    Name = '代码 + 文档 + 报告 + hf-cache'
    Why  = 'git clone 拿不到这些：本地工作副本在 .git/info/exclude，个人工作日志与 report_data/ 在 .gitignore，各任务手写 DOCX 也被排除；hf-cache 是 bge 三件套权重，必须同一份才能复现分数'
    ExcludeDirs = $RootExcludeDirs; ExcludeFiles = $RootExcludeFiles
})

[void]$Manifest.Add(@{
    Kind = 'tree'; Src = "$Src\data\chroma_db_4m"; Rel = 'data\chroma_db_4m'
    Name = '向量库 chroma_db_4m'
    Why  = '重建需要 merged_4m.parquet(15G) + 约 4.5 小时嵌入；直接搬 64.6G 更划算。只有 6 个文件，无小文件惩罚'
})

[void]$Manifest.Add(@{
    Kind = 'tree'; Src = "$Src\data\bm25_index_4m"; Rel = 'data\bm25_index_4m'
    Name = 'BM25 索引'
    Why  = '重建需要 data/chunks(42G)，搬 3.3G 比搬 42G 划算'
})

[void]$Manifest.Add(@{
    Kind = 'file'; Src = "$Src\data\docs_catalog.db"; Rel = 'data'
    Name = '文献目录 docs_catalog.db'
    Why  = '重建只要 45 秒，但需要 merged_4m.parquet(15G)。不搬 parquet 就必须搬它'
})

[void]$Manifest.Add(@{
    Kind = 'tree'; Src = "$Src\data\golden"; Rel = 'data\golden'
    Name = 'golden 检索评测集'
    Why  = '回归集，不可再生（已失去 held-out 资格，但仍是唯一一把无方差的尺子）'
})

[void]$Manifest.Add(@{
    Kind = 'tree'; Src = "$Src\data\landmark"; Rel = 'data\landmark'
    Name = 'landmark 条目'
    Why  = '全项目唯一一次联网请求的产物（10 个 PMID 的摘要原文）。搬过去 Mac 就永远不用联网'
})

[void]$Manifest.Add(@{
    Kind = 'tree'; Src = "$Src\data\chroma_landmark"; Rel = 'data\chroma_landmark'
    Name = 'landmark collection'
    Why  = '可由 entries.json 离线重建，但只有 1M，搬了省事'
})

[void]$Manifest.Add(@{
    Kind = 'tree'; Src = "$Src\data\dict"; Rel = 'data\dict'
    Name = '同义词词典 + 语料元数据'
    Why  = '重建需要 data/mesh 那份 299M XML；本体只有 10M，直接搬'
})

[void]$Manifest.Add(@{
    Kind = 'tree'; Src = "$Src\data\tokenizer"; Rel = 'data\tokenizer'
    Name = 'qwen3 分词器'
    Why  = '可从 Ollama GGUF 秒级重建，但只有 11M'
})

[void]$Manifest.Add(@{
    Kind = 'file'; Src = "$Src\data\index_stats.json"; Rel = 'data'
    Name = '索引统计'
    Why  = '建库时间等历史事实，重跑无法还原'
})

if ($IncludeParquet) {
    [void]$Manifest.Add(@{
        Kind = 'file'; Src = "$Src\data\vectors\merged_4m.parquet"; Rel = 'data\vectors'
        Name = '建库保险文件 merged_4m.parquet'
        Why  = '阶段四库损坏就是靠它恢复的。搬了 Mac 才能独立重建 chroma 库与文献目录'
    })
}

if ($IncludeCorpus) {
    [void]$Manifest.Add(@{
        Kind = 'tree'; Src = "$Src\data\chunks"; Rel = 'data\chunks'
        Name = '切块表 chunks'
        Why  = '-IncludeCorpus：重建 BM25 索引与重新抽样才需要'
    })
    [void]$Manifest.Add(@{
        Kind = 'tree'; Src = "$Src\data\pubmed"; Rel = 'data\pubmed'
        Name = '原始语料 pubmed'
        Why  = '-IncludeCorpus：只有重跑切块管线才需要'
    })
}

# ── 统计 ─────────────────────────────────────────────────────────────────────
function Get-ItemSize {
    param($Item)
    if ($Item.Kind -eq 'file') {
        if (-not (Test-Path $Item.Src)) { return -1 }
        return (Get-Item $Item.Src).Length
    }
    if (-not (Test-Path $Item.Src)) { return -1 }

    # ⚠ 顶层剪枝，不能「先全量递归再过滤」：那样会去遍历 data/pubmed(76G) 与
    #   data/chunks(42G) 的整棵目录树，光统计就要等很久，而它们根本不在清单里。
    $ex = @()
    if ($Item.ExcludeDirs) { $ex = $Item.ExcludeDirs }
    $sum = 0L
    foreach ($child in Get-ChildItem $Item.Src -Force -ErrorAction SilentlyContinue) {
        $skip = $false
        foreach ($xd in $ex) {
            if ($child.FullName.Equals($xd, 'OrdinalIgnoreCase')) { $skip = $true; break }
        }
        if ($skip) { continue }
        if ($child.PSIsContainer) {
            $s = (Get-ChildItem $child.FullName -Recurse -Force -File -ErrorAction SilentlyContinue |
                  Measure-Object Length -Sum).Sum
            if ($s) { $sum += $s }
        } else {
            $sum += $child.Length
        }
    }
    return $sum
}

Write-Host ""
Write-Host "源: $Src"
Write-Host "目标: $Dest"
Write-Host ("=" * 78)

$total = 0L
$missing = @()
foreach ($item in $Manifest) {
    $sz = Get-ItemSize $item
    if ($sz -lt 0) {
        $missing += $item.Name
        Write-Host ("  {0,10}  {1}  << 源不存在，将跳过" -f '缺失', $item.Name) -ForegroundColor Yellow
        $item.Size = -1
        continue
    }
    $item.Size = $sz
    $total += $sz
    Write-Host ("  {0,10:N2} GB  {1}" -f ($sz / 1GB), $item.Name)
    Write-Host ("              └ $($item.Why)") -ForegroundColor DarkGray
}

Write-Host ("=" * 78)
Write-Host ("  合计 {0:N2} GB（{1} 项）" -f ($total / 1GB), ($Manifest.Count - $missing.Count))

# 时间估算：E: 在 NVMe SSD 上，磁盘不是瓶颈，只按链路算
$gb = $total / 1GB
Write-Host ""
Write-Host "  传输时间估（只算网络，实际 SMB 效率约理论值的 70~80%）："
Write-Host ("    2.5G 有线（本机网卡就是 2.5GbE，Mac 需转接器 + 交换机也要 2.5G）  约 {0:N0}~{1:N0} 分钟" -f ($gb / 16.8), ($gb / 12.0))
Write-Host ("    千兆有线                                                        约 {0:N0}~{1:N0} 分钟" -f ($gb / 6.7),  ($gb / 4.8))
Write-Host ("    Wi-Fi 6                                                         约 {0:N0}~{1:N0} 分钟（波动大）" -f ($gb / 6.0), ($gb / 2.0))

if ($DryRun) {
    Write-Host ""
    Write-Host "  -DryRun：以上为预演，未复制任何文件。" -ForegroundColor Cyan
    exit 0
}

# ── 目标可写性检查（在烧掉几十分钟之前先失败） ────────────────────────────────
if (-not (Test-Path $Dest)) {
    try { New-Item -ItemType Directory -Path $Dest -Force | Out-Null }
    catch { throw "目标不可达或不可创建: $Dest`n  Mac 侧要先开「文件共享」并共享一个可写文件夹，见 docs\迁移到Mac.md" }
}
$probe = Join-Path $Dest ('.write_probe_' + [guid]::NewGuid().ToString('N') + '.tmp')
try {
    Set-Content -Path $probe -Value 'probe' -Encoding utf8 -ErrorAction Stop
    Remove-Item $probe -Force -ErrorAction Stop
} catch {
    throw "目标不可写: $Dest`n  $($_.Exception.Message)"
}
Write-Host ""
Write-Host "  目标可写性检查通过。" -ForegroundColor Green

# ── 传输 ─────────────────────────────────────────────────────────────────────
# /Z    大文件断点续传（52G 的 chroma.sqlite3 靠它）
# /COPY:DT /DCOPY:T  只带数据与时间戳，不带 NTFS 属性（推到 macOS 共享上属性会失败）
# /R:2 /W:5          失败重试 2 次、间隔 5 秒，不要默认的一百万次
# /XO                目标更新则跳过，重跑安全
$common = @('/E', '/Z', '/COPY:DT', '/DCOPY:T', '/R:2', '/W:5', '/XO', '/NP', '/TEE', "/LOG+:$LogFile")

$results = @()
$idx = 0
foreach ($item in $Manifest) {
    $idx++
    if ($item.Size -lt 0) { continue }

    $target = $Dest
    if ($item.Rel -ne '') { $target = Join-Path $Dest $item.Rel }

    Write-Host ""
    Write-Host ("[{0}/{1}] {2}  ({3:N2} GB)" -f $idx, $Manifest.Count, $item.Name, ($item.Size / 1GB)) -ForegroundColor Cyan

    # ⚠ 不能用 $args 当变量名，那是 PowerShell 的自动变量
    if ($item.Kind -eq 'file') {
        $srcDir  = Split-Path $item.Src -Parent
        $srcFile = Split-Path $item.Src -Leaf
        $rcArgs = @($srcDir, $target, $srcFile) + ($common | Where-Object { $_ -ne '/E' })
    } else {
        $rcArgs = @($item.Src, $target) + $common
        if ($item.ExcludeDirs)  { $rcArgs += '/XD'; $rcArgs += $item.ExcludeDirs }
        if ($item.ExcludeFiles) { $rcArgs += '/XF'; $rcArgs += $item.ExcludeFiles }
    }

    $t0 = Get-Date
    & robocopy.exe @rcArgs
    $code = $LASTEXITCODE
    $dt = (Get-Date) - $t0

    # robocopy 退出码：<8 正常（0 无需复制 / 1 已复制 / 2 有多余项 / 3=1+2），>=8 出错
    $ok = ($code -lt 8)
    $rate = 0
    if ($dt.TotalSeconds -gt 0) { $rate = ($item.Size / 1MB) / $dt.TotalSeconds }
    $results += [pscustomobject]@{
        项目 = $item.Name; 退出码 = $code; 成功 = $ok
        用时 = ('{0:N1} 分' -f $dt.TotalMinutes); 速率 = ('{0:N0} MB/s' -f $rate)
    }
    if ($ok) { Write-Host ("  完成（退出码 $code，{0:N1} 分，{1:N0} MB/s）" -f $dt.TotalMinutes, $rate) -ForegroundColor Green }
    else     { Write-Host  "  失败（退出码 $code），详见 $LogFile" -ForegroundColor Red }
}

Write-Host ""
Write-Host ("=" * 78)
$results | Format-Table -AutoSize

$failed = @($results | Where-Object { -not $_.成功 })
$okCount = ($results | Where-Object { $_.成功 }).Count
Write-Host ("  成功 {0}/{1} 项；日志 {2}" -f $okCount, $results.Count, $LogFile)

if ($failed.Count -gt 0) {
    Write-Host ("  有 {0} 项失败，重跑本脚本会跳过已完成的文件、续传断掉的大文件。" -f $failed.Count) -ForegroundColor Red
    exit 1
}
if ($results.Count -eq 0) {
    Write-Host "  没有任何项被传输——清单为空或源全部缺失，这不是成功。" -ForegroundColor Red
    exit 1
}
Write-Host "  全部完成。下一步见 docs\迁移到Mac.md 第二节（Mac 侧重建）。" -ForegroundColor Green
exit 0
