$ErrorActionPreference = "Stop"

Write-Host "Starting Philosophical Engine demo..."

$pythonExe = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment python not found at $pythonExe"
}

$apiArgs = @("-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "8000")
$apiProcess = Start-Process -FilePath $pythonExe -ArgumentList $apiArgs -WorkingDirectory (Join-Path $PSScriptRoot "..") -PassThru -WindowStyle Hidden

try {
    $healthUrl = "http://127.0.0.1:8000/health"
    $maxAttempts = 30
    $ready = $false

    for ($i = 0; $i -lt $maxAttempts; $i++) {
        try {
            $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 2
            if ($health.status -eq "ok") {
                $ready = $true
                break
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }

    if (-not $ready) {
        throw "API did not become ready in time."
    }

    $predictPayload = @{
        text = "Virtue, reason, and discipline guide the individual toward tranquility."
    } | ConvertTo-Json

    $result = Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $predictPayload
    Write-Host ""
    Write-Host "Prediction response:"
    $result | ConvertTo-Json -Depth 5
}
finally {
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id -Force
    }
    Write-Host ""
    Write-Host "Demo finished. API server stopped."
}
