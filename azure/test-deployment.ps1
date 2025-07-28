# ML Playground Deployment Status Check
param(
    [string]$ResourceGroup = "ml-playground"
)

Write-Host "🚀 ML Playground Deployment Status Check" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Cyan

# Test API
Write-Host "🔧 Testing API..." -ForegroundColor Cyan
try {
    $apiResponse = Invoke-RestMethod -Uri "https://ml-playground-api.azurewebsites.net/" -TimeoutSec 10
    Write-Host "✅ API is working: $($apiResponse.message)" -ForegroundColor Green
    
    # Test API docs
    try {
        $docsResponse = Invoke-WebRequest -Uri "https://ml-playground-api.azurewebsites.net/docs" -Method Head -TimeoutSec 10
        Write-Host "✅ API Docs accessible (Status: $($docsResponse.StatusCode))" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ API Docs: $($_.Exception.Message)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ API Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Test Frontend
Write-Host "🎨 Testing Frontend..." -ForegroundColor Cyan
try {
    $frontendResponse = Invoke-WebRequest -Uri "https://ml-playground-frontend.azurewebsites.net" -Method Head -TimeoutSec 30
    Write-Host "✅ Frontend is accessible (Status: $($frontendResponse.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Frontend: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "   Note: Streamlit apps may take 30-60 seconds to start up" -ForegroundColor Gray
}

Write-Host ""
Write-Host "🌐 Your Application URLs:" -ForegroundColor Magenta
Write-Host "📡 API: https://ml-playground-api.azurewebsites.net" -ForegroundColor Cyan
Write-Host "📖 API Documentation: https://ml-playground-api.azurewebsites.net/docs" -ForegroundColor Cyan
Write-Host "🎨 Frontend: https://ml-playground-frontend.azurewebsites.net" -ForegroundColor Cyan

Write-Host ""
Write-Host "🎯 Next Steps for Custom Domain:" -ForegroundColor Yellow
Write-Host "1. Run: .\azure\setup-domain.ps1" -ForegroundColor White
Write-Host "2. Add DNS records to vanshdeshwal.dev:" -ForegroundColor White
Write-Host "   CNAME: playground → ml-playground-frontend.azurewebsites.net" -ForegroundColor Gray
Write-Host "3. Final URL: https://playground.vanshdeshwal.dev" -ForegroundColor Green

Write-Host ""
Write-Host "💰 Cost: FREE (F1 tier) - Perfect for Azure for Students!" -ForegroundColor Green
Write-Host "⚡ Note: First load may take 30-60 seconds (cold start)" -ForegroundColor Gray
