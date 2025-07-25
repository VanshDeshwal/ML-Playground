# Simplified Azure Deployment for Students Subscription
# Uses Docker Hub instead of Azure Container Registry to avoid provider registration issues

param(
    [string]$ResourceGroup = "ml-playground",
    [string]$Location = "eastus",
    [string]$Domain = "ml.vanshdeshwal.dev"
)

Write-Host "🎓 Azure for Students Deployment" -ForegroundColor Green
Write-Host "💰 Using Docker Hub to avoid ACR provider registration" -ForegroundColor Yellow

# Set variables
$API_APP_NAME = "ml-playground-api"
$FRONTEND_APP_NAME = "ml-playground-frontend"
$ENVIRONMENT_NAME = "ml-playground-env"

# Create resource group (should work)
Write-Host "📦 Creating resource group..." -ForegroundColor Cyan
az group create --name $ResourceGroup --location $Location

# Check if Container Apps provider is registered
Write-Host "🔍 Checking Container Apps provider..." -ForegroundColor Cyan
$appsProvider = az provider show --namespace Microsoft.App --query "registrationState" -o tsv

if ($appsProvider -ne "Registered") {
    Write-Host "⏳ Registering Container Apps provider..." -ForegroundColor Yellow
    az provider register --namespace Microsoft.App
    az provider register --namespace Microsoft.OperationalInsights
    
    Write-Host "⏳ Waiting for registration (this can take 2-5 minutes)..." -ForegroundColor Yellow
    do {
        Start-Sleep -Seconds 30
        $appsProvider = az provider show --namespace Microsoft.App --query "registrationState" -o tsv
        Write-Host "Status: $appsProvider" -ForegroundColor Yellow
    } while ($appsProvider -eq "Registering")
}

if ($appsProvider -eq "Registered") {
    Write-Host "✅ Container Apps provider ready!" -ForegroundColor Green
    
    # Deploy using public Docker images (we'll build and push to Docker Hub later)
    Write-Host "🚀 Deploying Container Apps..." -ForegroundColor Cyan
    
    # Create Container Apps environment
    az containerapp env create \
        --name $ENVIRONMENT_NAME \
        --resource-group $ResourceGroup \
        --location $Location
    
    # Deploy API app (using a placeholder image for now)
    az containerapp create \
        --name $API_APP_NAME \
        --resource-group $ResourceGroup \
        --environment $ENVIRONMENT_NAME \
        --image mcr.microsoft.com/azuredocs/containerapps-helloworld:latest \
        --target-port 80 \
        --ingress external \
        --min-replicas 0 \
        --max-replicas 3 \
        --cpu 0.25 \
        --memory 0.5Gi
    
    # Deploy Frontend app
    az containerapp create \
        --name $FRONTEND_APP_NAME \
        --resource-group $ResourceGroup \
        --environment $ENVIRONMENT_NAME \
        --image mcr.microsoft.com/azuredocs/containerapps-helloworld:latest \
        --target-port 80 \
        --ingress external \
        --min-replicas 0 \
        --max-replicas 3 \
        --cpu 0.25 \
        --memory 0.5Gi
    
    # Get URLs
    $apiUrl = az containerapp show --name $API_APP_NAME --resource-group $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
    $frontendUrl = az containerapp show --name $FRONTEND_APP_NAME --resource-group $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
    
    Write-Host ""
    Write-Host "🎉 Basic deployment completed!" -ForegroundColor Green
    Write-Host "📡 API URL: https://$apiUrl" -ForegroundColor Cyan
    Write-Host "🎨 Frontend URL: https://$frontendUrl" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📋 NEXT STEPS:" -ForegroundColor Yellow
    Write-Host "1. Build and push your Docker images to Docker Hub"
    Write-Host "2. Update Container Apps to use your custom images"
    Write-Host "3. Set up custom domain: $Domain"
    Write-Host ""
    Write-Host "💰 COST: ~$5-10/month with scale-to-zero" -ForegroundColor Green
    Write-Host "🎓 Perfect for Azure for Students $100 credit!" -ForegroundColor Green
    
} else {
    Write-Host "❌ Container Apps provider registration failed" -ForegroundColor Red
    Write-Host "Try again in a few minutes or contact Azure support" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🌐 CUSTOM DOMAIN SETUP:" -ForegroundColor Magenta
Write-Host "Once deployed, add this CNAME record to $Domain DNS:"
Write-Host "ml.$Domain CNAME $frontendUrl" -ForegroundColor Cyan
