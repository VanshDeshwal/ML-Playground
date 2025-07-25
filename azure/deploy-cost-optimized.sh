#!/bin/bash

# Cost-Optimized Azure ML Playground Deployment
# Uses Container Apps with scale-to-zero for maximum cost savings

set -e

# Configuration
RESOURCE_GROUP="ml-playground-cost-optimized"
LOCATION="eastus"  # Cheapest region
REGISTRY_NAME="mlplaygroundreg$(date +%s | tail -c 6)"  # Unique short name
ENVIRONMENT_NAME="ml-playground-env"
API_APP_NAME="ml-playground-api"
FRONTEND_APP_NAME="ml-playground-frontend"

echo "💰 Starting COST-OPTIMIZED ML Playground deployment..."
echo "🎯 Target: ~$5-15/month (vs $40-55 with Container Instances)"

# Create resource group
echo "📦 Creating resource group in cheapest region..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create minimal Azure Container Registry (Basic tier)
echo "🐳 Creating minimal Container Registry..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $REGISTRY_NAME \
    --sku Basic \
    --admin-enabled true

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP --query "loginServer" --output tsv)
echo "📡 ACR Login Server: $ACR_LOGIN_SERVER"

# Build and push optimized images
echo "🔨 Building OPTIMIZED API image..."
cd ../backend
az acr build --registry $REGISTRY_NAME --image ml-playground-api:latest .

echo "🎨 Building OPTIMIZED Frontend image..."
cd ../frontend  
az acr build --registry $REGISTRY_NAME --image ml-playground-frontend:latest .

# Deploy using Container Apps (scale-to-zero)
echo "🚀 Deploying with Container Apps (scale-to-zero)..."
cd ../azure

az deployment group create \
    --resource-group $RESOURCE_GROUP \
    --template-file container-apps-deployment.json \
    --parameters \
        environmentName=$ENVIRONMENT_NAME \
        apiAppName=$API_APP_NAME \
        frontendAppName=$FRONTEND_APP_NAME \
        apiImage="${ACR_LOGIN_SERVER}/ml-playground-api:latest" \
        frontendImage="${ACR_LOGIN_SERVER}/ml-playground-frontend:latest"

# Get URLs
API_URL=$(az deployment group show \
    --resource-group $RESOURCE_GROUP \
    --name container-apps-deployment \
    --query properties.outputs.apiUrl.value \
    --output tsv)

FRONTEND_URL=$(az deployment group show \
    --resource-group $RESOURCE_GROUP \
    --name container-apps-deployment \
    --query properties.outputs.frontendUrl.value \
    --output tsv)

echo ""
echo "🎉 COST-OPTIMIZED deployment completed!"
echo ""
echo "📡 API URL: $API_URL"
echo "🎨 Frontend URL: $FRONTEND_URL"
echo ""
echo "💰 COST SAVINGS FEATURES:"
echo "   ✅ Scale-to-zero (pay only when used)"
echo "   ✅ Minimal CPU/Memory allocation (0.25 vCPU, 0.5GB)"
echo "   ✅ Basic tier services"
echo "   ✅ Cheapest region (East US)"
echo "   ✅ 30-day log retention only"
echo ""
echo "💡 ESTIMATED COSTS:"
echo "   📊 Container Apps: ~$2-8/month (scale-to-zero)"
echo "   🐳 Container Registry: ~$5/month (Basic)"
echo "   📝 Log Analytics: ~$2-5/month (minimal logs)"
echo "   🎯 TOTAL: ~$9-18/month (vs $40-55 with ACI)"
echo ""
echo "⚡ PERFORMANCE:"
echo "   🚀 Cold start: ~10-30 seconds (first request after idle)"
echo "   ⚡ Warm requests: <1 second"
echo "   🔄 Auto-scales based on demand"
echo ""
echo "🔧 To further reduce costs:"
echo "   - Delete when not in use: az group delete --name $RESOURCE_GROUP"
echo "   - Use only during development/demos"
echo "   - Consider serverless alternatives for production"
