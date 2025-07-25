# 🌐 ML Portfolio Domain Strategy for vanshdeshwal.dev

## 🎯 **Current Project: Interactive ML Playground**
**Subdomain**: `playground.vanshdeshwal.dev`
**Description**: Interactive platform for experimenting with various ML algorithms (Linear Regression, Clustering, Neural Networks)

## 🏗️ **Complete Portfolio Architecture**

### **Main Portfolio Site**
```
vanshdeshwal.dev
├── About/Resume
├── Project Showcase
├── Blog/Articles
└── Contact
```

### **ML Project Subdomains**
```
1. playground.vanshdeshwal.dev    # ✅ Current: Interactive ML Playground
   - Linear Regression
   - Clustering (K-means, Blobs)
   - Neural Networks
   - Algorithm Comparisons

2. credit.vanshdeshwal.dev        # 🔄 Next: Credit Risk Modeling
   - Risk Assessment Models
   - Credit Scoring
   - Default Prediction
   - Portfolio Analysis

3. timeseries.vanshdeshwal.dev    # 📈 Future: Time Series Analysis
   - Stock Price Prediction
   - Sales Forecasting
   - Seasonal Analysis
   - ARIMA/LSTM Models

4. nlp.vanshdeshwal.dev          # 📝 Future: Natural Language Processing
   - Sentiment Analysis
   - Text Classification
   - Named Entity Recognition
   - Chatbot/Q&A Systems

5. vision.vanshdeshwal.dev       # 👁️ Future: Computer Vision
   - Image Classification
   - Object Detection
   - Face Recognition
   - Medical Image Analysis

6. api.vanshdeshwal.dev          # 🔧 Optional: Unified API Gateway
   - Centralized ML APIs
   - Authentication
   - Rate Limiting
   - Documentation Hub
```

## 🎨 **DNS Configuration Template**

### **For playground.vanshdeshwal.dev (Current)**
```dns
Type: CNAME
Name: playground
Target: ml-playground-frontend.azurewebsites.net
TTL: 300

Type: TXT
Name: asuid.playground
Value: [verification-id]
TTL: 300
```

### **Future Project Template**
```dns
# Credit Risk Project
Type: CNAME
Name: credit
Target: credit-risk-app.azurewebsites.net

# Time Series Project  
Type: CNAME
Name: timeseries
Target: timeseries-app.azurewebsites.net
```

## 📊 **Cost Planning with Azure for Students**

### **Free Tier Strategy (Recommended)**
```
Per Project:
- Web App (F1): $0/month
- App Service Plan: Shared across projects
- SSL Certificates: $0/month (automatic)

Total for 3-5 projects: $0/month
```

### **Scalable Tier (If needed later)**
```
Per Project:
- Web App (B1): ~$13/month
- Shared resources: Database, Storage
- Custom domains: $0/month

Total for 3 projects: ~$40/month
```

## 🚀 **Deployment Strategy**

### **Current Project Setup**
```powershell
# 1. Deploy playground.vanshdeshwal.dev
az deployment group create \
  --resource-group ml-playground \
  --template-file webapp-deployment.json

# 2. Configure custom domain
.\setup-domain.ps1 -Domain "playground.vanshdeshwal.dev"
```

### **Future Project Template**
```powershell
# Credit Risk Project
az group create --name credit-risk-ml --location eastus

az deployment group create \
  --resource-group credit-risk-ml \
  --template-file webapp-deployment.json \
  --parameters \
    apiAppName="credit-risk-api" \
    frontendAppName="credit-risk-frontend" \
    gitRepoUrl="https://github.com/VanshDeshwal/Credit-Risk-ML.git"
```

## 🎯 **Professional Branding Benefits**

### **Portfolio Visitors Will See:**
1. **playground.vanshdeshwal.dev** - "Interactive ML experimentation platform"
2. **credit.vanshdeshwal.dev** - "Production-ready credit risk assessment"
3. **timeseries.vanshdeshwal.dev** - "Advanced forecasting and prediction models"

### **SEO & Professional Impact:**
- Each subdomain targets specific ML keywords
- Demonstrates specialization and depth
- Easy to share specific projects with employers
- Professional email signatures: "Check out my ML playground at playground.vanshdeshwal.dev"

## 🔗 **Cross-Project Integration**

### **Navigation Between Projects**
```html
<!-- Add to each project's header -->
<nav class="ml-portfolio-nav">
  <a href="https://vanshdeshwal.dev">Home</a>
  <a href="https://playground.vanshdeshwal.dev">ML Playground</a>
  <a href="https://credit.vanshdeshwal.dev">Credit Risk</a>
  <a href="https://timeseries.vanshdeshwal.dev">Time Series</a>
</nav>
```

### **Unified Analytics**
- Google Analytics across all subdomains
- Track visitor flow between projects
- Measure engagement per ML domain

## 📈 **Growth Roadmap**

### **Phase 1 (Current): Foundation**
- ✅ playground.vanshdeshwal.dev deployed
- Set up domain infrastructure
- Establish deployment patterns

### **Phase 2 (Next 2-3 months): Expansion**
- 🔄 Deploy credit.vanshdeshwal.dev
- Create consistent UI/branding
- Add cross-project navigation

### **Phase 3 (Future): Specialization**
- Add timeseries.vanshdeshwal.dev
- Consider api.vanshdeshwal.dev for unified access
- Add advanced features (user accounts, saved models)

## 🎨 **Branding Consistency**

### **Color Scheme Template**
```css
:root {
  --primary: #2E86AB;      /* Professional blue */
  --secondary: #A23B72;    /* Accent purple */
  --success: #F18F01;      /* Orange for CTAs */
  --background: #F5F5F5;   /* Light gray */
  --text: #2D3436;         /* Dark gray */
}
```

### **Project-Specific Accents**
- **Playground**: Blue/Green (experimentation)
- **Credit**: Red/Orange (risk/finance)
- **TimeSeries**: Purple/Blue (data/prediction)

**Recommendation: Use `playground.vanshdeshwal.dev` for this project - it's specific, professional, and sets up a perfect foundation for your ML portfolio expansion!**
