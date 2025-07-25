# Check ML Playground Deployment Status
param(
    [string]$ResourceGroup = "ml-playground"
)

Write-Host "🔍 Checking ML Playground deployment status..." -ForegroundColor Cyan

# Check if resource group exists
$rgExists = az group exists --name $ResourceGroup
if ($rgExists -eq "false") {
    Write-Host "❌ Resource group '$ResourceGroup' not found" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Resource group exists" -ForegroundColor Green

# Check App Service Plan
Write-Host "📋 App Service Plan status:" -ForegroundColor Cyan
az appservice plan show --name "ml-playground-plan" --resource-group $ResourceGroup --query "{name:name,sku:sku.name,status:status}" --output table

# Check API App
Write-Host "🔧 API App status:" -ForegroundColor Cyan
$apiStatus = az webapp show --name "ml-playground-api" --resource-group $ResourceGroup --query "{name:name,state:state,hostNames:defaultHostName}" --output table
Write-Host $apiStatus

# Check Frontend App  
Write-Host "🎨 Frontend App status:" -ForegroundColor Cyan
$frontendStatus = az webapp show --name "ml-playground-frontend" --resource-group $ResourceGroup --query "{name:name,state:state,hostNames:defaultHostName}" --output table
Write-Host $frontendStatus

# Get URLs
Write-Host "🌐 Application URLs:" -ForegroundColor Green
$apiUrl = az webapp show --name "ml-playground-api" --resource-group $ResourceGroup --query "defaultHostName" --output tsv
$frontendUrl = az webapp show --name "ml-playground-frontend" --resource-group $ResourceGroup --query "defaultHostName" --output tsv

Write-Host "📡 API: https://$apiUrl" -ForegroundColor Cyan
Write-Host "📡 API Docs: https://$apiUrl/docs" -ForegroundColor Cyan  
Write-Host "🎨 Frontend: https://$frontendUrl" -ForegroundColor Cyan

Write-Host ""
Write-Host "💡 Next steps:" -ForegroundColor Yellow
Write-Host "1. Wait for deployment to complete (~5-10 minutes)"
Write-Host "2. Test API: https://$apiUrl/docs"
Write-Host "3. Access frontend: https://$frontendUrl"
Write-Host "4. Set up custom domain: ml.vanshdeshwal.dev"

Write-Host ""
Write-Host "💰 Cost: FREE TIER (F1) - $0/month!" -ForegroundColor Green
Write-Host "🎯 Perfect for Azure for Students subscription" -ForegroundColor Green
