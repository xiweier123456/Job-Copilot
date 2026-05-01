$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

docker compose -f "$Root\infra\docker-compose.yml" down
