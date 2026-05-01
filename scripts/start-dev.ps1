param(
    [string]$BackendPython = "C:\Users\A\anaconda3\envs\demo_01\python.exe",
    [string]$FrontendDir = "",
    [int]$BackendPort = 8001,
    [int]$McpPort = 8000,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($FrontendDir)) {
    $FrontendDir = Join-Path $Root "vue"
}

Write-Host "Starting Redis..." -ForegroundColor Cyan
docker compose -f "$Root\infra\redis\docker-compose.yml" up -d

Write-Host "Starting FastAPI backend on http://127.0.0.1:$BackendPort ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd `"$Root`"; `$env:DEBUG='false'; & `"$BackendPython`" -m uvicorn app.main:app --host 127.0.0.1 --port $BackendPort --reload"
)

Write-Host "Starting MCP server on http://127.0.0.1:$McpPort/mcp ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd `"$Root`"; `$env:MCP_HOST='127.0.0.1'; `$env:MCP_PORT='$McpPort'; `$env:MCP_PATH='/mcp'; `$env:MCP_TRANSPORT='streamable-http'; & `"$BackendPython`" -m app.mcp.server"
)

Write-Host "Starting Vue frontend on http://127.0.0.1:$FrontendPort ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd `"$FrontendDir`"; npm run dev -- --host 127.0.0.1 --port $FrontendPort"
)

Write-Host ""
Write-Host "Dev stack is starting:" -ForegroundColor Green
Write-Host "  Backend : http://127.0.0.1:$BackendPort"
Write-Host "  MCP     : http://127.0.0.1:$McpPort/mcp"
Write-Host "  Frontend: http://127.0.0.1:$FrontendPort"
Write-Host "  Redis   : redis://127.0.0.1:6379/0"
