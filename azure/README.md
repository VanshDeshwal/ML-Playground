# 🚀 Azure Deployment Guide for ML Playground

This guide walks you through deploying the ML Playground application to Azure using containerized microservices.

## 📋 Prerequisites

1. **Azure CLI**: Install and configure Azure CLI
   ```bash
   # Install Azure CLI (Windows)
   winget install Microsoft.AzureCLI
   
   # Install Azure CLI (macOS)
   brew install azure-cli
   
   # Install Azure CLI (Linux)
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   ```

2. **Azure Subscription**: Ensure you have an active Azure subscription

3. **Login to Azure**:
   ```bash
   az login
   ```

## 🏗️ Architecture Overview

The deployment consists of two separate containers:

- **API Container**: FastAPI backend with ML algorithms
- **Frontend Container**: Streamlit web interface

Both containers are deployed to Azure Container Instances (ACI) with public endpoints.

## 🚀 Quick Deployment

### Option 1: Automated Script (Recommended)

**Linux/macOS:**
```bash
cd azure
chmod +x deploy.sh
./deploy.sh
```

**Windows PowerShell:**
```powershell
cd azure
.\deploy.ps1
```

### Option 2: Manual Deployment

1. **Create Resource Group**:
   ```bash
   az group create --name ml-playground-rg --location eastus
   ```

2. **Create Container Registry**:
   ```bash
   az acr create --resource-group ml-playground-rg --name mlplaygroundregistry --sku Basic --admin-enabled true
   ```

3. **Build and Push Images**:
   ```bash
   # API Image
   cd backend
   az acr build --registry mlplaygroundregistry --image ml-playground-api:latest .
   
   # Frontend Image
   cd ../frontend
   az acr build --registry mlplaygroundregistry --image ml-playground-frontend:latest .
   ```

4. **Deploy API Container**:
   ```bash
   cd ../azure
   az deployment group create \
     --resource-group ml-playground-rg \
     --template-file api-deployment.json \
     --parameters \
       containerImage="mlplaygroundregistry.azurecr.io/ml-playground-api:latest" \
       dnsNameLabel="ml-playground-api-unique"
   ```

5. **Deploy Frontend Container**:
   ```bash
   az deployment group create \
     --resource-group ml-playground-rg \
     --template-file frontend-deployment.json \
     --parameters \
       containerImage="mlplaygroundregistry.azurecr.io/ml-playground-frontend:latest" \
       apiBaseUrl="http://YOUR-API-FQDN:8000" \
       dnsNameLabel="ml-playground-frontend-unique"
   ```

## 🧪 Local Testing with Docker

Before deploying to Azure, test locally:

```bash
# Build and run both containers
docker-compose up --build

# Access the application
# Frontend: http://localhost:8501
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## 🔧 Configuration

### Environment Variables

**Frontend Container:**
- `API_BASE_URL`: URL of the API service (automatically configured during deployment)

**API Container:**
- No additional environment variables required

### Resource Requirements

**Default Resource Allocation:**
- **CPU**: 1 core per container
- **Memory**: 2 GB per container

To modify resources, edit the ARM templates in the `azure/` directory.

## 📊 Monitoring and Logs

### View Container Logs

```bash
# API logs
az container logs --resource-group ml-playground-rg --name ml-playground-api

# Frontend logs  
az container logs --resource-group ml-playground-rg --name ml-playground-frontend
```

### Health Checks

Both containers include health check endpoints:
- **API**: `http://your-api-url:8000/health`
- **Frontend**: `http://your-frontend-url:8501/_stcore/health`

## 🚨 Troubleshooting

### Common Issues

1. **Container Fails to Start**:
   ```bash
   # Check container events
   az container show --resource-group ml-playground-rg --name ml-playground-api
   ```

2. **API Connection Issues**:
   - Verify the API URL is correctly set in frontend environment variables
   - Check if the API container is running and healthy

3. **Resource Limits**:
   - Increase CPU/memory allocation in ARM templates if containers are slow

### Cleanup Resources

```bash
# Delete entire resource group (removes all resources)
az group delete --name ml-playground-rg --yes --no-wait
```

## 💰 Cost Optimization

### Azure Container Instances Pricing

- **CPU**: ~$0.0012 per vCPU per hour
- **Memory**: ~$0.00016 per GB per hour

**Estimated Monthly Cost**: ~$35-50 USD for both containers running 24/7

### Cost Saving Tips

1. **Scale Down**: Stop containers when not in use
2. **Optimize Resources**: Reduce CPU/memory if sufficient
3. **Use Spot Instances**: Consider Azure Container Groups with spot pricing

## 🔄 Updates and CI/CD

### Manual Updates

```bash
# Rebuild and redeploy API
cd backend
az acr build --registry mlplaygroundregistry --image ml-playground-api:latest .

# Restart container instance
az container restart --resource-group ml-playground-rg --name ml-playground-api
```

### Automated CI/CD

Consider setting up GitHub Actions or Azure DevOps for automated deployments:

1. **Trigger**: Push to main branch
2. **Build**: Create new container images
3. **Deploy**: Update container instances
4. **Test**: Run health checks

## 📞 Support

For deployment issues:

1. Check container logs using Azure CLI
2. Verify all prerequisites are met
3. Ensure Azure subscription has sufficient quotas
4. Review ARM template parameters

## 🎯 Next Steps

After successful deployment:

1. **Custom Domain**: Configure custom domain name
2. **SSL/HTTPS**: Add SSL certificate for secure access  
3. **Backup**: Implement backup strategy for data
4. **Monitoring**: Set up Azure Monitor for alerts
5. **Scaling**: Consider Azure Kubernetes Service (AKS) for higher scale

Your ML Playground is now running in the cloud! 🎉
