# 🌐 Custom Domain Setup for ML Playground

## 🎯 **Recommended Subdomain Options for vanshdeshwal.dev:**

### **🥇 Best Options:**
1. **`ml.vanshdeshwal.dev`** - Clean, professional, memorable
2. **`playground.vanshdeshwal.dev`** - Descriptive and fun
3. **`ai.vanshdeshwal.dev`** - Short, modern, AI-focused

### **🥈 Alternative Options:**
4. **`mllab.vanshdeshwal.dev`** - Lab concept, experimentation focus
5. **`models.vanshdeshwal.dev`** - ML models showcase
6. **`datascience.vanshdeshwal.dev`** - Full data science scope

### **🎨 Creative Options:**
7. **`neural.vanshdeshwal.dev`** - Neural networks reference
8. **`predict.vanshdeshwal.dev`** - Prediction/analytics focus
9. **`insights.vanshdeshwal.dev`** - Data insights platform

## 🚀 **Deployment Steps with Custom Domain**

### **Phase 1: Deploy to Azure (Current)**
```powershell
# 1. Create resource group
az group create --name ml-playground --location eastus

# 2. Create container registry  
az acr create --resource-group ml-playground --name $REGISTRY_NAME --sku Basic --admin-enabled true

# 3. Build and push images
az acr build --registry $REGISTRY_NAME --image ml-playground-api:latest ../backend
az acr build --registry $REGISTRY_NAME --image ml-playground-frontend:latest ../frontend

# 4. Deploy Container Apps with custom domain support
az deployment group create --resource-group ml-playground --template-file container-apps-deployment.json
```

### **Phase 2: Configure Custom Domain**

#### **Step 1: Get Azure URLs (after deployment)**
```bash
# API URL (example)
https://ml-playground-api.happyfield-12345.eastus.azurecontainerapps.io

# Frontend URL (example)  
https://ml-playground-frontend.happyfield-12345.eastus.azurecontainerapps.io
```

#### **Step 2: DNS Configuration**
Add these CNAME records to your vanshdeshwal.dev DNS:

```dns
# For ml.vanshdeshwal.dev
ml.vanshdeshwal.dev     CNAME   ml-playground-frontend.happyfield-12345.eastus.azurecontainerapps.io

# For API subdomain (optional)
api.vanshdeshwal.dev    CNAME   ml-playground-api.happyfield-12345.eastus.azurecontainerapps.io
```

#### **Step 3: SSL Certificate (Automatic)**
Azure Container Apps will automatically provision SSL certificates for your custom domain.

### **Phase 3: Domain Binding Commands**

```bash
# 1. Add custom domain to Container App
az containerapp hostname add \
  --hostname ml.vanshdeshwal.dev \
  --resource-group ml-playground \
  --name ml-playground-frontend

# 2. Bind SSL certificate (automatic)
az containerapp hostname bind \
  --hostname ml.vanshdeshwal.dev \
  --resource-group ml-playground \
  --name ml-playground-frontend \
  --validation-method CNAME
```

## 🎯 **Final Architecture**

### **Production URLs:**
- **Frontend**: `https://ml.vanshdeshwal.dev`
- **API**: `https://ml.vanshdeshwal.dev/api` (or separate `api.vanshdeshwal.dev`)
- **API Docs**: `https://ml.vanshdeshwal.dev/api/docs`

### **Cost with Azure for Students:**
- **Container Apps**: $2-8/month (scale-to-zero)
- **Container Registry**: $5/month (Basic tier)
- **Domain SSL**: $0 (automatic with Container Apps)
- **Total**: ~$7-13/month (within your $100 credit!)

## 🔧 **DNS Provider Setup Examples**

### **Cloudflare:**
```
Type: CNAME
Name: ml
Target: ml-playground-frontend.happyfield-12345.eastus.azurecontainerapps.io
TTL: Auto
Proxy: Off (important for Container Apps)
```

### **Namecheap:**
```
Type: CNAME Record
Host: ml
Value: ml-playground-frontend.happyfield-12345.eastus.azurecontainerapps.io
TTL: Automatic
```

### **GoDaddy:**
```
Type: CNAME
Name: ml
Value: ml-playground-frontend.happyfield-12345.eastus.azurecontainerapps.io
TTL: 1 Hour
```

## ⚡ **Quick Setup Script (After Deployment)**

```powershell
# Variables (replace with your actual values)
$DOMAIN = "ml.vanshdeshwal.dev"
$RESOURCE_GROUP = "ml-playground"
$APP_NAME = "ml-playground-frontend"

# Add and bind custom domain
az containerapp hostname add --hostname $DOMAIN --resource-group $RESOURCE_GROUP --name $APP_NAME
az containerapp hostname bind --hostname $DOMAIN --resource-group $RESOURCE_GROUP --name $APP_NAME --validation-method CNAME

Write-Host "✅ Domain configured! Update your DNS with the CNAME record." -ForegroundColor Green
```

## 🎨 **Professional Landing Page Ideas**

Once deployed to `ml.vanshdeshwal.dev`, consider these sections:

1. **Hero Section**: "Vansh's ML Playground - Explore Machine Learning Models"
2. **Live Demos**: Interactive model testing
3. **Algorithm Showcase**: Linear Regression, Clustering, Neural Networks
4. **Portfolio Integration**: Link to your other projects
5. **GitHub Integration**: Direct links to source code
6. **Resume/CV Link**: Professional development showcase

## 🌟 **SEO Optimization**
```html
<title>ML Playground - Vansh Deshwal | Machine Learning Portfolio</title>
<meta name="description" content="Interactive machine learning playground featuring regression, classification, and clustering algorithms. Built by Vansh Deshwal.">
<meta name="keywords" content="machine learning, AI, data science, portfolio, Vansh Deshwal">
```

## 📈 **Analytics Setup**
Add Google Analytics or Plausible to track:
- Model usage patterns
- Popular algorithms
- Geographic reach
- Portfolio engagement

**Recommendation: Use `ml.vanshdeshwal.dev` - it's professional, memorable, and perfect for your ML portfolio!**
