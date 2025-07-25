# Cost-Optimized Azure ML Playground Deployment (PowerShell)
# Uses Container Apps with scale-to-zero for maximum cost savings

# Configuration
$RESOURCE_GROUP = "ml-playground-cost-optimized"
$LOCATION = "eastus"  # Cheapest region
$RANDOM_SUFFIX = Get-Random -Minimum 100000 -Maximum 999999
$REGISTRY_NAME = "mlplaygroundreg$RANDOM_SUFFIX"  # Unique name
$ENVIRONMENT_NAME = "ml-playground-env"
$API_APP_NAME = "ml-playground-api"
$FRONTEND_APP_NAME = "ml-playground-frontend"

Write-Host "💰 Starting COST-OPTIMIZED ML Playground deployment..." -ForegroundColor Green
Write-Host "🎯 Target: ~$5-15/month (vs $40-55 with Container Instances)" -ForegroundColor Yellow

# Check if Azure CLI is available
try {
    $null = az --version 2>&1
    Write-Host "✅ Azure CLI found" -ForegroundColor Green
} catch {
    Write-Host "❌ Azure CLI not found. Please install it first:" -ForegroundColor Red
    Write-Host "   winget install Microsoft.AzureCLI" -ForegroundColor Yellow
    Write-Host "   Then restart PowerShell and run this script again." -ForegroundColor Yellow
    exit 1
}

# Check if logged in to Azure
try {
    $account = az account show 2>&1 | ConvertFrom-Json
    Write-Host "✅ Logged in as: $($account.user.name)" -ForegroundColor Green
} catch {
    Write-Host "🔐 Please log in to Azure first:" -ForegroundColor Yellow
    az login
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Azure login failed. Please try again." -ForegroundColor Red
        exit 1
    }
}

# Create resource group
Write-Host "📦 Creating resource group in cheapest region..." -ForegroundColor Cyan
az group create --name $RESOURCE_GROUP --location $LOCATION
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create resource group" -ForegroundColor Red
    exit 1
}

# Create minimal Azure Container Registry (Basic tier)
Write-Host "🐳 Creating minimal Container Registry..." -ForegroundColor Cyan
az acr create `
    --resource-group $RESOURCE_GROUP `
    --name $REGISTRY_NAME `
    --sku Basic `
    --admin-enabled true

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create Container Registry" -ForegroundColor Red
    exit 1
}

# Get ACR login server
$ACR_LOGIN_SERVER = az acr show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP --query "loginServer" --output tsv
Write-Host "📡 ACR Login Server: $ACR_LOGIN_SERVER" -ForegroundColor Green

# Build and push optimized images
Write-Host "🔨 Building OPTIMIZED API image..." -ForegroundColor Cyan
Set-Location "../backend"
az acr build --registry $REGISTRY_NAME --image ml-playground-api:latest .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to build API image" -ForegroundColor Red
    exit 1
}

Write-Host "🎨 Building OPTIMIZED Frontend image..." -ForegroundColor Cyan
Set-Location "../frontend"
az acr build --registry $REGISTRY_NAME --image ml-playground-frontend:latest .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to build Frontend image" -ForegroundColor Red
    exit 1
}

# Deploy using Container Apps (scale-to-zero)
Write-Host "🚀 Deploying with Container Apps (scale-to-zero)..." -ForegroundColor Cyan
Set-Location "../azure"

az deployment group create `
    --resource-group $RESOURCE_GROUP `
    --template-file container-apps-deployment.json `
    --parameters `
        environmentName=$ENVIRONMENT_NAME `
        apiAppName=$API_APP_NAME `
        frontendAppName=$FRONTEND_APP_NAME `
        apiImage="$ACR_LOGIN_SERVER/ml-playground-api:latest" `
        frontendImage="$ACR_LOGIN_SERVER/ml-playground-frontend:latest"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to deploy Container Apps" -ForegroundColor Red
    exit 1
}

# Get URLs
$API_URL = az deployment group show `
    --resource-group $RESOURCE_GROUP `
    --name container-apps-deployment `
    --query properties.outputs.apiUrl.value `
    --output tsv

$FRONTEND_URL = az deployment group show `
    --resource-group $RESOURCE_GROUP `
    --name container-apps-deployment `
    --query properties.outputs.frontendUrl.value `
    --output tsv

Write-Host ""
Write-Host "🎉 COST-OPTIMIZED deployment completed!" -ForegroundColor Green
Write-Host ""
Write-Host "📡 API URL: $API_URL" -ForegroundColor Cyan
Write-Host "🎨 Frontend URL: $FRONTEND_URL" -ForegroundColor Cyan
Write-Host ""
Write-Host "💰 COST SAVINGS FEATURES:" -ForegroundColor Yellow
Write-Host "   ✅ Scale-to-zero (pay only when used)"
Write-Host "   ✅ Minimal CPU/Memory allocation (0.25 vCPU, 0.5GB)"
Write-Host "   ✅ Basic tier services"
Write-Host "   ✅ Cheapest region (East US)"
Write-Host "   ✅ 30-day log retention only"
Write-Host ""
Write-Host "💡 ESTIMATED COSTS:" -ForegroundColor Green
Write-Host "   📊 Container Apps: ~$2-8/month (scale-to-zero)"
Write-Host "   🐳 Container Registry: ~$5/month (Basic)"
Write-Host "   📝 Log Analytics: ~$2-5/month (minimal logs)"
Write-Host "   🎯 TOTAL: ~$9-18/month (vs $40-55 with ACI)" -ForegroundColor Green
Write-Host ""
Write-Host "⚡ PERFORMANCE:" -ForegroundColor Cyan
Write-Host "   🚀 Cold start: ~10-30 seconds (first request after idle)"
Write-Host "   ⚡ Warm requests: under 1 second"
Write-Host "   🔄 Auto-scales based on demand"
Write-Host ""
Write-Host "🔧 To further reduce costs:" -ForegroundColor Yellow
Write-Host "   - Delete when not in use: az group delete --name $RESOURCE_GROUP"
Write-Host "   - Use only during development/demos"
Write-Host "   - Consider serverless alternatives for production"
Write-Host ""
Write-Host "🎯 Next steps:" -ForegroundColor Magenta
Write-Host "   1. Test your APIs: $API_URL/docs"
Write-Host "   2. Access frontend: $FRONTEND_URL"
Write-Host "   3. Set up budget alerts (see COST-OPTIMIZATION.md)"
Write-Host "   4. Monitor costs in Azure portal"
