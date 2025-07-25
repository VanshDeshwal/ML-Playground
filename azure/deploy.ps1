# Azure ML Playground Deployment Script (PowerShell)
# This script deploys both API and Frontend containers to Azure Container Instances

param(
    [string]$ResourceGroup = "ml-playground-rg",
    [string]$Location = "eastus",
    [string]$RegistryName = "mlplaygroundregistry"
)

$ErrorActionPreference = "Stop"

# Configuration
$ApiImageName = "ml-playground-api"
$FrontendImageName = "ml-playground-frontend"
$ApiDnsLabel = "ml-playground-api-$(Get-Date -Format 'yyyyMMddHHmmss')"
$FrontendDnsLabel = "ml-playground-frontend-$(Get-Date -Format 'yyyyMMddHHmmss')"

Write-Host "🚀 Starting ML Playground deployment to Azure..." -ForegroundColor Green

# Create resource group
Write-Host "📦 Creating resource group..." -ForegroundColor Yellow
az group create --name $ResourceGroup --location $Location

# Create Azure Container Registry
Write-Host "🐳 Creating Azure Container Registry..." -ForegroundColor Yellow
az acr create --resource-group $ResourceGroup --name $RegistryName --sku Basic --admin-enabled true

# Get ACR login server
$AcrLoginServer = az acr show --name $RegistryName --resource-group $ResourceGroup --query "loginServer" --output tsv
Write-Host "📡 ACR Login Server: $AcrLoginServer" -ForegroundColor Cyan

# Build and push API image
Write-Host "🔨 Building and pushing API image..." -ForegroundColor Yellow
Set-Location "../backend"
az acr build --registry $RegistryName --image "${ApiImageName}:latest" .

# Build and push Frontend image
Write-Host "🎨 Building and pushing Frontend image..." -ForegroundColor Yellow
Set-Location "../frontend"
az acr build --registry $RegistryName --image "${FrontendImageName}:latest" .

# Deploy API container
Write-Host "🚀 Deploying API container..." -ForegroundColor Yellow
Set-Location "../azure"
az deployment group create `
    --resource-group $ResourceGroup `
    --template-file api-deployment.json `
    --parameters `
        containerImage="${AcrLoginServer}/${ApiImageName}:latest" `
        dnsNameLabel=$ApiDnsLabel

# Get API URL
$ApiFqdn = az deployment group show `
    --resource-group $ResourceGroup `
    --name api-deployment `
    --query properties.outputs.containerFQDN.value `
    --output tsv

$ApiBaseUrl = "http://${ApiFqdn}:8000"
Write-Host "📡 API deployed at: $ApiBaseUrl" -ForegroundColor Green

# Deploy Frontend container
Write-Host "🎨 Deploying Frontend container..." -ForegroundColor Yellow
az deployment group create `
    --resource-group $ResourceGroup `
    --template-file frontend-deployment.json `
    --parameters `
        containerImage="${AcrLoginServer}/${FrontendImageName}:latest" `
        apiBaseUrl=$ApiBaseUrl `
        dnsNameLabel=$FrontendDnsLabel

# Get Frontend URL
$FrontendFqdn = az deployment group show `
    --resource-group $ResourceGroup `
    --name frontend-deployment `
    --query properties.outputs.containerFQDN.value `
    --output tsv

$FrontendUrl = "http://${FrontendFqdn}:8501"
Write-Host "🎨 Frontend deployed at: $FrontendUrl" -ForegroundColor Green

Write-Host ""
Write-Host "🎉 Deployment completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📡 API Endpoint: $ApiBaseUrl" -ForegroundColor Cyan
Write-Host "🎨 Frontend URL: $FrontendUrl" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 You can now access your ML Playground at the frontend URL" -ForegroundColor Yellow
Write-Host "📖 API documentation available at: ${ApiBaseUrl}/docs" -ForegroundColor Yellow
Write-Host ""
Write-Host "🔧 To manage your deployment:" -ForegroundColor Yellow
Write-Host "   - Resource Group: $ResourceGroup" -ForegroundColor White
Write-Host "   - Container Registry: $RegistryName" -ForegroundColor White
Write-Host "   - Location: $Location" -ForegroundColor White
