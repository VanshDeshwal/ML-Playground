# Quick Health Check for ML Playground Apps
param(
    [string]$ResourceGroup = "ml-playground"
)

Write-Host "🏥 ML Playground Health Check" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

# Get URLs
$apiUrl = "https://ml-playground-api.azurewebsites.net"
$frontendUrl = "https://ml-playground-frontend.azurewebsites.net"

Write-Host "📡 Testing API: $apiUrl" -ForegroundColor Cyan
try {
    $apiResponse = Invoke-WebRequest -Uri "$apiUrl/health" -TimeoutSec 30 -ErrorAction Stop
    Write-Host "✅ API Health: $($apiResponse.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "❌ API Health: Failed - $($_.Exception.Message)" -ForegroundColor Red
    
    # Try root endpoint
    try {
        $rootResponse = Invoke-WebRequest -Uri "$apiUrl/" -TimeoutSec 30 -ErrorAction Stop
        Write-Host "✅ API Root: $($rootResponse.StatusCode)" -ForegroundColor Green
    } catch {
        Write-Host "❌ API Root: Failed - $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "🎨 Testing Frontend: $frontendUrl" -ForegroundColor Cyan
try {
    $frontendResponse = Invoke-WebRequest -Uri $frontendUrl -TimeoutSec 30 -ErrorAction Stop
    Write-Host "✅ Frontend: $($frontendResponse.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "❌ Frontend: Failed - $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "🔗 Direct Links:" -ForegroundColor Yellow
Write-Host "API Docs: $apiUrl/docs"
Write-Host "API Health: $apiUrl/health"
Write-Host "Frontend: $frontendUrl"
Write-Host ""
Write-Host "🛠️ Troubleshooting:" -ForegroundColor Yellow
Write-Host "If apps are failing, check logs with:"
Write-Host "az webapp log tail --resource-group $ResourceGroup --name ml-playground-api"
Write-Host "az webapp log tail --resource-group $ResourceGroup --name ml-playground-frontend"
