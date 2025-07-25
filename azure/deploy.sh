#!/bin/bash

# Azure ML Playground Deployment Script
# This script deploys both API and Frontend containers to Azure Container Instances

set -e

# Configuration
RESOURCE_GROUP="ml-playground-rg"
LOCATION="eastus"
REGISTRY_NAME="mlplaygroundregistry"
API_IMAGE_NAME="ml-playground-api"
FRONTEND_IMAGE_NAME="ml-playground-frontend"
API_DNS_LABEL="ml-playground-api-$(date +%s)"
FRONTEND_DNS_LABEL="ml-playground-frontend-$(date +%s)"

echo "🚀 Starting ML Playground deployment to Azure..."

# Create resource group
echo "📦 Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
echo "🐳 Creating Azure Container Registry..."
az acr create --resource-group $RESOURCE_GROUP --name $REGISTRY_NAME --sku Basic --admin-enabled true

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP --query "loginServer" --output tsv)
echo "📡 ACR Login Server: $ACR_LOGIN_SERVER"

# Build and push API image
echo "🔨 Building and pushing API image..."
cd ../backend
az acr build --registry $REGISTRY_NAME --image ${API_IMAGE_NAME}:latest .

# Build and push Frontend image  
echo "🎨 Building and pushing Frontend image..."
cd ../frontend
az acr build --registry $REGISTRY_NAME --image ${FRONTEND_IMAGE_NAME}:latest .

# Deploy API container
echo "🚀 Deploying API container..."
cd ../azure
az deployment group create \
    --resource-group $RESOURCE_GROUP \
    --template-file api-deployment.json \
    --parameters \
        containerImage="${ACR_LOGIN_SERVER}/${API_IMAGE_NAME}:latest" \
        dnsNameLabel=$API_DNS_LABEL

# Get API URL
API_FQDN=$(az deployment group show \
    --resource-group $RESOURCE_GROUP \
    --name api-deployment \
    --query properties.outputs.containerFQDN.value \
    --output tsv)

API_BASE_URL="http://${API_FQDN}:8000"
echo "📡 API deployed at: $API_BASE_URL"

# Deploy Frontend container
echo "🎨 Deploying Frontend container..."
az deployment group create \
    --resource-group $RESOURCE_GROUP \
    --template-file frontend-deployment.json \
    --parameters \
        containerImage="${ACR_LOGIN_SERVER}/${FRONTEND_IMAGE_NAME}:latest" \
        apiBaseUrl=$API_BASE_URL \
        dnsNameLabel=$FRONTEND_DNS_LABEL

# Get Frontend URL
FRONTEND_FQDN=$(az deployment group show \
    --resource-group $RESOURCE_GROUP \
    --name frontend-deployment \
    --query properties.outputs.containerFQDN.value \
    --output tsv)

FRONTEND_URL="http://${FRONTEND_FQDN}:8501"
echo "🎨 Frontend deployed at: $FRONTEND_URL"

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📡 API Endpoint: $API_BASE_URL"
echo "🎨 Frontend URL: $FRONTEND_URL"
echo ""
echo "💡 You can now access your ML Playground at the frontend URL"
echo "📖 API documentation available at: ${API_BASE_URL}/docs"
echo ""
echo "🔧 To manage your deployment:"
echo "   - Resource Group: $RESOURCE_GROUP"
echo "   - Container Registry: $REGISTRY_NAME"
echo "   - Location: $LOCATION"
