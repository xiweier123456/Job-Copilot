param(
    [Parameter(Mandatory = $true)]
    [string]$Message,
    [string]$Branch = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $Branch) {
    $Branch = git branch --show-current
}

if (-not $Branch) {
    throw "Cannot detect current git branch."
}

Write-Host "Running tests before sync..." -ForegroundColor Cyan
& "C:\Users\A\anaconda3\envs\demo_01\python.exe" -m pytest -q

Write-Host "Pulling latest changes from origin/$Branch ..." -ForegroundColor Cyan
git pull --rebase origin $Branch

Write-Host "Committing local changes..." -ForegroundColor Cyan
git add -A
git commit -m $Message

Write-Host "Pushing to origin/$Branch ..." -ForegroundColor Cyan
git push origin $Branch
