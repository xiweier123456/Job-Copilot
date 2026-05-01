param(
    [string]$FrontendDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($FrontendDir)) {
    $FrontendDir = Join-Path $Root "vue"
}

Write-Host "Stopping standalone Redis compose, if it exists..." -ForegroundColor Cyan
docker compose -f "$Root\infra\redis\docker-compose.yml" down

Write-Host "Starting full Docker stack..." -ForegroundColor Cyan
$env:FRONTEND_DIR = $FrontendDir
docker compose -f "$Root\infra\docker-compose.yml" up -d --build

Write-Host ""
Write-Host "Docker stack is starting:" -ForegroundColor Green
Write-Host "  Frontend: http://127.0.0.1:5173"
Write-Host "  Backend : http://127.0.0.1:8001"
Write-Host "  MCP     : http://127.0.0.1:8000/mcp"
Write-Host "  Milvus  : 127.0.0.1:19530"
Write-Host "  Redis   : redis://127.0.0.1:6379/0"
Write-Host ""
Write-Host "Logs:"
Write-Host "  docker compose -f infra\docker-compose.yml logs -f backend"
Write-Host "  docker compose -f infra\docker-compose.yml logs -f mcp"
