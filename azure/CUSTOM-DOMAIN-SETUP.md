# 🌐 Custom Domain Setup for ML Playground
# Optimized for Azure for Students ($100/year credit)

## 🎯 Perfect Subdomain Suggestions for `vanshdeshwal.dev`

### 🥇 **Recommended Options:**

1. **`ml.vanshdeshwal.dev`** ⭐ **BEST CHOICE**
   - Short, memorable, professional
   - Clear ML/Machine Learning focus
   - Perfect for showcasing to employers/peers

2. **`playground.vanshdeshwal.dev`** 
   - Descriptive of the interactive nature
   - Inviting and approachable
   - Great for educational content

3. **`lab.vanshdeshwal.dev`**
   - Professional, research-oriented feel
   - Perfect for experimentation platform
   - Short and easy to remember

### 🥈 **Alternative Options:**

4. **`demo.vanshdeshwal.dev`** - For showcasing capabilities
5. **`learn.vanshdeshwal.dev`** - Educational focus
6. **`ai.vanshdeshwal.dev`** - Broader AI/ML scope
7. **`data.vanshdeshwal.dev`** - Data science focus
8. **`models.vanshdeshwal.dev`** - ML models focus

---

## 💰 Cost Optimization with Azure for Students

### **Your Advantage:**
- ✅ **$100 free credits annually**
- ✅ **No credit card required**
- ✅ **Access to free services**
- ✅ **Educational pricing on paid services**

### **Projected Costs with Container Apps:**
```
Monthly Breakdown:
📊 Container Apps: $2-8/month (scale-to-zero)
🐳 Container Registry: $5/month (Basic)
📝 Log Analytics: $2-5/month
🌐 Custom Domain: $0 (free with Container Apps)
📧 Budget Alerts: $0 (free)

💰 TOTAL: ~$9-18/month
🎯 Annual Cost: ~$108-216
✅ Covered by $100 student credit + ~$8-116 out of pocket
```

### **Ultra-Low Cost Strategy (Recommended for Students):**
```
🔄 Development Mode: Use start/stop scripts
⏰ Run 4 hours/day average = ~$6/month
📅 Annual cost: ~$72 (completely covered by credits!)
💡 Perfect for: Learning, demos, portfolio projects
```

---

## 🚀 Custom Domain Setup Steps

### **Option 1: Free Subdomain (Recommended)**
```bash
# After deployment, get your Container App URL
$FRONTEND_URL = az deployment group show \
  --resource-group ml-playground-cost-optimized \
  --name container-apps-deployment \
  --query properties.outputs.frontendUrl.value \
  --output tsv

# Create CNAME record in your domain DNS:
# ml.vanshdeshwal.dev -> $FRONTEND_URL
```

### **Option 2: Custom Domain with SSL (Free with Container Apps)**
```bash
# Add custom domain to Container App
az containerapp hostname add \
  --resource-group ml-playground-cost-optimized \
  --name ml-playground-frontend \
  --hostname ml.vanshdeshwal.dev

# Verify domain and get SSL certificate (automatic)
az containerapp hostname verify \
  --resource-group ml-playground-cost-optimized \
  --name ml-playground-frontend \
  --hostname ml.vanshdeshwal.dev
```

---

## 📋 DNS Configuration

### **For CloudFlare (if using):**
```
Type: CNAME
Name: ml
Target: <your-container-app-url>
TTL: Auto
Proxy: Off (Orange cloud disabled)
```

### **For Other DNS Providers:**
```
Record Type: CNAME
Host/Name: ml
Points to: <your-container-app-url>
TTL: 300 seconds
```

---

## 🔧 Post-Deployment Domain Setup

### **1. Get Your Container App URLs:**
```powershell
# Get frontend URL
$FRONTEND_URL = az deployment group show `
  --resource-group ml-playground-cost-optimized `
  --name container-apps-deployment `
  --query properties.outputs.frontendUrl.value `
  --output tsv

# Get API URL  
$API_URL = az deployment group show `
  --resource-group ml-playground-cost-optimized `
  --name container-apps-deployment `
  --query properties.outputs.apiUrl.value `
  --output tsv

Write-Host "Frontend: $FRONTEND_URL"
Write-Host "API: $API_URL"
```

### **2. Update DNS Records:**
```
CNAME Records to Add:
- ml.vanshdeshwal.dev -> $FRONTEND_URL
- api.vanshdeshwal.dev -> $API_URL (optional)
```

### **3. Enable Custom Domain:**
```bash
# Add frontend custom domain
az containerapp hostname add \
  --resource-group ml-playground-cost-optimized \
  --name ml-playground-frontend \
  --hostname ml.vanshdeshwal.dev

# Add API custom domain (optional)
az containerapp hostname add \
  --resource-group ml-playground-cost-optimized \
  --name ml-playground-api \
  --hostname api.vanshdeshwal.dev
```

### **4. Verify and Get SSL:**
```bash
# Verify frontend domain (gets free SSL)
az containerapp hostname verify \
  --resource-group ml-playground-cost-optimized \
  --name ml-playground-frontend \
  --hostname ml.vanshdeshwal.dev

# Check SSL certificate status
az containerapp hostname list \
  --resource-group ml-playground-cost-optimized \
  --name ml-playground-frontend
```

---

## 🎯 Final Setup Results

After setup, you'll have:
- ✅ **`ml.vanshdeshwal.dev`** - Your ML Playground
- ✅ **Free SSL certificate** (auto-renewed)
- ✅ **Professional domain** for portfolio
- ✅ **Scale-to-zero** cost optimization
- ✅ **Covered by student credits**

---

## 💡 Pro Tips for Students

### **Portfolio Enhancement:**
1. Add to resume/CV as live project
2. Include in GitHub README with live link
3. Share with professors/classmates
4. Use for internship applications

### **Cost Management:**
1. Set $20/month budget alert
2. Use development start/stop workflow
3. Delete when not actively using
4. Monitor costs weekly

### **Domain Strategy:**
1. Start with subdomain (free)
2. Keep main domain for personal site
3. Consider `lab.vanshdeshwal.dev` for future projects
4. Use `api.vanshdeshwal.dev` for backend showcase

---

## 🚨 Important Notes

### **DNS Propagation:**
- Changes take 5-60 minutes globally
- Test with `nslookup ml.vanshdeshwal.dev`
- Use incognito/private browsing for testing

### **SSL Certificate:**
- Automatic with Container Apps
- Takes 5-15 minutes after domain verification
- Shows green lock in browser when ready

### **Student Credit Management:**
- Monitor usage in Azure portal
- Set up billing alerts
- Plan projects around credit renewal

Your ML Playground will be accessible at **`https://ml.vanshdeshwal.dev`** - perfect for your portfolio! 🎉
