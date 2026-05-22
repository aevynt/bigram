Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

powershell -ExecutionPolicy Bypass -File scripts\windows\setup_server.ps1
powershell -ExecutionPolicy Bypass -File scripts\windows\prepare_tensor1.ps1

Write-Host ""
Write-Host "Train:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\windows\train_tensor1_48gb.ps1"
Write-Host "Server:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\windows\run_server.ps1"
Write-Host "Smoke test:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\windows\smoke_test.ps1"
