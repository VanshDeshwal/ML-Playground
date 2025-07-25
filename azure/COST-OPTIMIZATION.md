# 💰 Cost Optimization Guide for ML Playground

This comprehensive guide shows you how to reduce Azure costs from **$40-55/month** down to **$2-15/month** or even **near-zero** for development use.

## 🎯 Cost Reduction Strategies (Ranked by Impact)

### 🥇 **Option 1: Container Apps with Scale-to-Zero** 
**💰 Cost: $2-15/month (vs $40-55)**

```bash
cd azure
chmod +x deploy-cost-optimized.sh
./deploy-cost-optimized.sh
```

**Benefits:**
- ✅ Automatically scales to zero when idle (no usage = no cost)
- ✅ Minimal resource allocation (0.25 vCPU, 0.5GB RAM)
- ✅ Built-in HTTPS and load balancing
- ✅ Cold start: ~10-30 seconds (acceptable for demos)

**Best for:** Development, demos, low-traffic usage

---

### 🥈 **Option 2: Start/Stop Control Script**
**💰 Cost: $4-50/month (depending on usage)**

```bash
# Start when needed
./cost-control.sh start

# Stop when done (saves ~$1.40/day)
./cost-control.sh stop

# Check status and costs
./cost-control.sh status
./cost-control.sh cost
```

**Savings Examples:**
- 8 hours/day: ~$12/month (74% savings)
- 4 hours/day: ~$6/month (87% savings)
- Weekends only: ~$3/month (94% savings)

**Best for:** Scheduled usage, development cycles

---

### 🥉 **Option 3: Serverless with Azure Functions**
**💰 Cost: $0-5/month (pay-per-request)**

```bash
cd azure/serverless
# Deploy simplified ML endpoints as Functions
az functionapp create --consumption-plan-location eastus \
  --resource-group ml-playground-serverless \
  --name ml-playground-functions \
  --storage-account mlplaygroundstorage \
  --functions-version 4 \
  --runtime python
```

**Benefits:**
- ✅ True pay-per-use (first 1M requests free)
- ✅ No idle costs
- ✅ Auto-scaling
- ❌ Limited ML capabilities (simplified algorithms only)

**Best for:** Light usage, API-only scenarios

---

## 📊 Detailed Cost Comparison

| Strategy | Monthly Cost | Savings | Use Case |
|----------|--------------|---------|----------|
| **Original ACI** | $40-55 | 0% | 24/7 production |
| **Container Apps** | $2-15 | 70-96% | Development/demos |
| **Start/Stop Script** | $4-50 | 10-90% | Scheduled usage |
| **Azure Functions** | $0-5 | 90-100% | Minimal usage |
| **Local Only** | $0 | 100% | Development only |

## 🔧 Resource Optimization Settings

### **Minimal Container Resources:**
```yaml
resources:
  cpu: 0.25      # Down from 1.0 vCPU
  memory: 0.5Gi  # Down from 2.0 GB
```

### **Minimal Docker Images:**
- Use `Dockerfile.minimal` (Alpine Linux)
- Essential packages only (`requirements-minimal.txt`)
- ~200MB vs ~800MB original

### **Region Selection:**
- **Cheapest**: East US, South Central US
- **Avoid**: West Europe, North Europe (20-30% more expensive)

## 🚨 Cost Monitoring & Alerts

### **Set Budget Alerts:**
```bash
# Set $20/month budget with email alerts
az deployment group create \
  --resource-group ml-playground-rg \
  --template-file budget-alert.json \
  --parameters \
    amount=20 \
    contactEmail="your-email@domain.com" \
    resourceGroupId="/subscriptions/YOUR-SUB-ID/resourceGroups/ml-playground-rg"
```

### **Daily Cost Monitoring:**
```bash
# Check current month spending
az consumption usage list \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --resource-group ml-playground-rg
```

## 🎮 Development Workflow for Maximum Savings

### **Option A: Container Apps (Recommended)**
1. Deploy once with `deploy-cost-optimized.sh`
2. Apps automatically scale to zero when idle
3. Cold start on first request (~30 seconds)
4. Warm requests are fast (<1 second)

### **Option B: Start/Stop Workflow**
```bash
# Morning - start work
./cost-control.sh start
# Wait 2-3 minutes for containers to start
# Work on ML Playground

# Evening - stop work  
./cost-control.sh stop
# Costs reduced to ~$0.15/day for storage only
```

### **Option C: Local Development + Cloud Demos**
```bash
# Daily development - local only
python run.py

# Demo/sharing - deploy temporarily
./deploy-cost-optimized.sh
# After demo
az group delete --name ml-playground-cost-optimized
```

## 💡 Advanced Cost Hacks

### **1. Use Azure Free Tier**
- $200 free credits for new accounts
- Free services tier (limited resources)
- Perfect for learning/experimentation

### **2. Student Discounts**
- Azure for Students: $100 free credits
- GitHub Student Pack: Additional benefits
- Educational subscriptions: Reduced rates

### **3. Dev/Test Subscriptions**
- 40-60% discount on compute resources
- Available for MSDN subscribers
- Perfect for non-production workloads

### **4. Spot Instances**
```bash
# Use spot pricing for compute (up to 90% savings)
# Good for batch processing, training jobs
az container create --resource-group ml-playground-rg \
  --name ml-playground-spot \
  --image your-image \
  --priority Spot
```

### **5. Reserved Instances**
- 1-3 year commitments
- Up to 72% savings for predictable workloads
- Only if you plan to run 24/7

## 🏆 Recommended Setup by Use Case

### **Learning/Experimentation:**
```bash
# Local development + occasional cloud demos
Cost: $0-2/month
Strategy: Local + temporary cloud deployments
```

### **Portfolio/Demo Project:**
```bash
# Container Apps with scale-to-zero
Cost: $5-10/month  
Strategy: deploy-cost-optimized.sh
```

### **Active Development:**
```bash
# Start/stop workflow
Cost: $10-20/month
Strategy: cost-control.sh start/stop
```

### **Production (Small Scale):**
```bash
# Container Apps with monitoring
Cost: $15-30/month
Strategy: Container Apps + budget alerts
```

## 📈 Scaling Strategy

Start cheap and scale up as needed:

1. **Phase 1**: Local development ($0)
2. **Phase 2**: Container Apps ($5-10/month)
3. **Phase 3**: Dedicated containers ($20-30/month)
4. **Phase 4**: Kubernetes/AKS ($50+/month)

## ✅ Action Plan

**Immediate Actions (Next 5 Minutes):**
1. Deploy cost-optimized version: `./deploy-cost-optimized.sh`
2. Set budget alert: Use `budget-alert.json`
3. Bookmark cost monitoring commands

**Weekly Actions:**
1. Check spending: `az consumption usage list`
2. Review container logs for optimization opportunities
3. Clean up unused resources

**Monthly Actions:**
1. Review and optimize resource allocation
2. Consider scaling up/down based on usage
3. Evaluate new Azure cost optimization features

Your ML Playground can now run for **less than the cost of a Netflix subscription!** 🎉
