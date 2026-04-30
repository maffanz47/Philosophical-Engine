Write-Host "========================================="
Write-Host "   Starting The Philosophical Engine"
Write-Host "========================================="

# Activate virtual environment
if (Test-Path ".\venv\Scripts\activate.ps1") {
    Write-Host "Activating virtual environment..."
    . ".\venv\Scripts\activate.ps1"
} else {
    Write-Host "Virtual environment not found. Please ensure dependencies are installed."
}

# Ask to run pipeline
$runPipeline = Read-Host "Do you want to run the ML Pipeline to fetch data and train models? (y/n) [Default: n]"
if ($runPipeline -eq 'y' -or $runPipeline -eq 'Y') {
    Write-Host "Running Pipeline (this may take a few minutes)..."
    .\venv\Scripts\python.exe -m pipeline.philosophical_pipeline
}

# Start FastAPI server
Write-Host "Starting FastAPI web server..."
Write-Host "The Dashboard will be available at: http://localhost:8000/dashboard/"
Write-Host "Press Ctrl+C to stop the server."
.\venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000
