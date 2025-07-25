# Cost-Optimized Azure ML Playground Deployment
# Simple version that works with PowerShell

param(
    [string]$ResourceGroup = "ml-playground-cost-optimized",
    [string]$Location = "eastus"
)

Write-Host "💰 Starting COST-OPTIMIZED ML Playground deployment..." -ForegroundColor Green
Write-Host "🎯 Target: ~$5-15/month (vs $40-55 with Container Instances)" -ForegroundColor Yellow

# Generate unique names
$RANDOM_SUFFIX = Get-Random -Minimum 100000 -Maximum 999999
$REGISTRY_NAME = "mlplaygroundreg$RANDOM_SUFFIX"
$ENVIRONMENT_NAME = "ml-playground-env"
$API_APP_NAME = "ml-playground-api"
$FRONTEND_APP_NAME = "ml-playground-frontend"

Write-Host "📝 Configuration:" -ForegroundColor Cyan
Write-Host "   Resource Group: $ResourceGroup"
Write-Host "   Location: $Location"
Write-Host "   Registry: $REGISTRY_NAME"

# Step 1: Login to Azure
Write-Host "🔐 Checking Azure login..." -ForegroundColor Cyan
az account show
if ($LASTEXITCODE -ne 0) {
    Write-Host "Please log in to Azure:" -ForegroundColor Yellow
    az login
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Azure login failed" -ForegroundColor Red
        exit 1
    }
}

# Step 2: Create resource group
Write-Host "📦 Creating resource group..." -ForegroundColor Cyan
az group create --name $ResourceGroup --location $Location
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create resource group" -ForegroundColor Red
    exit 1
}

# Step 3: Create Container Registry
Write-Host "🐳 Creating Container Registry..." -ForegroundColor Cyan
az acr create --resource-group $ResourceGroup --name $REGISTRY_NAME --sku Basic --admin-enabled true
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create Container Registry" -ForegroundColor Red
    exit 1
}

# Step 4: Get ACR login server
Write-Host "📡 Getting ACR details..." -ForegroundColor Cyan
$ACR_LOGIN_SERVER = az acr show --name $REGISTRY_NAME --resource-group $ResourceGroup --query "loginServer" --output tsv
Write-Host "ACR Login Server: $ACR_LOGIN_SERVER" -ForegroundColor Green

# Step 5: Build and push images
Write-Host "🔨 Building API image..." -ForegroundColor Cyan
Set-Location "../backend"
az acr build --registry $REGISTRY_NAME --image ml-playground-api:latest .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to build API image" -ForegroundColor Red
    exit 1
}

Write-Host "🎨 Building Frontend image..." -ForegroundColor Cyan
Set-Location "../frontend"
az acr build --registry $REGISTRY_NAME --image ml-playground-frontend:latest .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to build Frontend image" -ForegroundColor Red
    exit 1
}

# Step 6: Deploy Container Apps
Write-Host "🚀 Deploying Container Apps..." -ForegroundColor Cyan
Set-Location "../azure"

az deployment group create `
    --resource-group $ResourceGroup `
    --template-file container-apps-deployment.json `
    --parameters `
        environmentName=$ENVIRONMENT_NAME `
        apiAppName=$API_APP_NAME `
        frontendAppName=$FRONTEND_APP_NAME `
        apiImage="$ACR_LOGIN_SERVER/ml-playground-api:latest" `
        frontendImage="$ACR_LOGIN_SERVER/ml-playground-frontend:latest"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Deployment failed" -ForegroundColor Red
    exit 1
}

# Step 7: Get deployment outputs
Write-Host "🔍 Getting deployment URLs..." -ForegroundColor Cyan
$API_URL = az deployment group show --resource-group $ResourceGroup --name container-apps-deployment --query properties.outputs.apiUrl.value --output tsv
$FRONTEND_URL = az deployment group show --resource-group $ResourceGroup --name container-apps-deployment --query properties.outputs.frontendUrl.value --output tsv

# Success!
Write-Host ""
Write-Host "🎉 DEPLOYMENT COMPLETED!" -ForegroundColor Green
Write-Host ""
Write-Host "📡 API URL: $API_URL" -ForegroundColor Cyan
Write-Host "🎨 Frontend URL: $FRONTEND_URL" -ForegroundColor Cyan
Write-Host ""
Write-Host "💰 COST SAVINGS:" -ForegroundColor Yellow
Write-Host "✅ Scale-to-zero when idle"
Write-Host "✅ Minimal resources (0.25 vCPU, 0.5GB RAM)"
Write-Host "✅ Basic tier services"
Write-Host "✅ East US region (cheapest)"
Write-Host ""
Write-Host "💡 ESTIMATED MONTHLY COST: $9-18" -ForegroundColor Green
Write-Host "(vs $40-55 with regular Container Instances)"
Write-Host ""
Write-Host "⚡ NEXT STEPS:" -ForegroundColor Magenta
Write-Host "1. Test API: $API_URL/docs"
Write-Host "2. Access frontend: $FRONTEND_URL" 
Write-Host "3. Set budget alerts (see COST-OPTIMIZATION.md)"
Write-Host ""
Write-Host "🔧 TO DELETE (when done):" -ForegroundColor Red
Write-Host "az group delete --name $ResourceGroup --yes --no-wait"
