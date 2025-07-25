# Custom Domain Setup for ML Playground
# Sets up ml.vanshdeshwal.dev to point to your Azure Web App

param(
    [string]$Domain = "playground.vanshdeshwal.dev",
    [string]$ResourceGroup = "ml-playground",
    [string]$AppName = "ml-playground-frontend"
)

Write-Host "🌐 Setting up custom domain: $Domain" -ForegroundColor Green

# Step 1: Get the current app hostname
$defaultHostname = az webapp show --name $AppName --resource-group $ResourceGroup --query "defaultHostName" --output tsv
Write-Host "📡 Current hostname: $defaultHostname" -ForegroundColor Cyan

# Step 2: Add custom domain
Write-Host "🔗 Adding custom domain..." -ForegroundColor Cyan
az webapp config hostname add --hostname $Domain --resource-group $ResourceGroup --webapp-name $AppName

# Step 3: Get domain verification ID
$verificationId = az webapp config hostname get-external-ip --name $AppName --resource-group $ResourceGroup
Write-Host "🔐 Domain verification ID: $verificationId" -ForegroundColor Yellow

Write-Host ""
Write-Host "📋 DNS CONFIGURATION REQUIRED:" -ForegroundColor Yellow
Write-Host "==============================================="
Write-Host "Add these DNS records to vanshdeshwal.dev:"
Write-Host ""
Write-Host "1. CNAME Record:" -ForegroundColor Cyan
Write-Host "   Type: CNAME"
Write-Host "   Name: playground"
Write-Host "   Value: $defaultHostname"
Write-Host "   TTL: 300 (or Auto)"
Write-Host ""
Write-Host "2. TXT Record (for verification):" -ForegroundColor Cyan
Write-Host "   Type: TXT"
Write-Host "   Name: asuid.playground"
Write-Host "   Value: $verificationId"
Write-Host "   TTL: 300 (or Auto)"
Write-Host ""
Write-Host "🎯 After adding DNS records, run:" -ForegroundColor Green
Write-Host "az webapp config ssl bind --certificate-thumbprint auto --ssl-type SNI --name $AppName --resource-group $ResourceGroup"
Write-Host ""
Write-Host "💡 Final URL will be: https://$Domain" -ForegroundColor Magenta

# Display current status
Write-Host ""
Write-Host "📊 Current Application Status:" -ForegroundColor Cyan
Write-Host "API URL: https://ml-playground-api.azurewebsites.net"
Write-Host "Frontend URL: https://ml-playground-frontend.azurewebsites.net"
Write-Host "API Docs: https://ml-playground-api.azurewebsites.net/docs"
Write-Host ""
Write-Host "💰 Cost: FREE (F1 tier) - Perfect for Azure for Students!" -ForegroundColor Green
