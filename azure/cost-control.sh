#!/bin/bash

# Development Mode - Start/Stop Script for Cost Control
# Only run containers when actively developing/demoing

RESOURCE_GROUP="ml-playground-rg"

show_help() {
    echo "💰 ML Playground Cost Control Script"
    echo ""
    echo "Usage: $0 [start|stop|status|cost]"
    echo ""
    echo "Commands:"
    echo "  start   - Start containers (begin billing)"
    echo "  stop    - Stop containers (pause billing)" 
    echo "  status  - Show current container status"
    echo "  cost    - Show estimated costs"
    echo ""
    echo "💡 Use this to control costs by only running when needed!"
}

start_containers() {
    echo "🚀 Starting ML Playground containers..."
    
    # Start API container
    echo "⚡ Starting API container..."
    az container start --resource-group $RESOURCE_GROUP --name ml-playground-api
    
    # Start Frontend container  
    echo "⚡ Starting Frontend container..."
    az container start --resource-group $RESOURCE_GROUP --name ml-playground-frontend
    
    echo "✅ Containers started! Billing resumed."
    echo "🌐 Check status with: $0 status"
}

stop_containers() {
    echo "🛑 Stopping ML Playground containers..."
    
    # Stop API container
    echo "💤 Stopping API container..."
    az container stop --resource-group $RESOURCE_GROUP --name ml-playground-api
    
    # Stop Frontend container
    echo "💤 Stopping Frontend container..."
    az container stop --resource-group $RESOURCE_GROUP --name ml-playground-frontend
    
    echo "✅ Containers stopped! Billing paused."
    echo "💰 You're now only paying for storage (~$5/month)"
}

show_status() {
    echo "📊 ML Playground Status:"
    echo ""
    
    # API status
    API_STATE=$(az container show --resource-group $RESOURCE_GROUP --name ml-playground-api --query "containers[0].instanceView.currentState.state" --output tsv 2>/dev/null || echo "Not Found")
    echo "📡 API Container: $API_STATE"
    
    # Frontend status  
    FRONTEND_STATE=$(az container show --resource-group $RESOURCE_GROUP --name ml-playground-frontend --query "containers[0].instanceView.currentState.state" --output tsv 2>/dev/null || echo "Not Found")
    echo "🎨 Frontend Container: $FRONTEND_STATE"
    
    if [[ "$API_STATE" == "Running" && "$FRONTEND_STATE" == "Running" ]]; then
        echo ""
        echo "💰 Status: BILLING ACTIVE (~$1.50/day)"
        echo "🌐 Your app is accessible and incurring costs"
        
        # Get URLs
        API_FQDN=$(az container show --resource-group $RESOURCE_GROUP --name ml-playground-api --query "ipAddress.fqdn" --output tsv 2>/dev/null || echo "N/A")
        FRONTEND_FQDN=$(az container show --resource-group $RESOURCE_GROUP --name ml-playground-frontend --query "ipAddress.fqdn" --output tsv 2>/dev/null || echo "N/A")
        
        echo "📡 API: http://$API_FQDN:8000"
        echo "🎨 Frontend: http://$FRONTEND_FQDN:8501"
        
    elif [[ "$API_STATE" == "Terminated" && "$FRONTEND_STATE" == "Terminated" ]]; then
        echo ""
        echo "💤 Status: PAUSED (~$0.15/day for storage only)"
        echo "🛑 Containers stopped - minimal costs"
        
    else
        echo ""
        echo "⚠️  Status: MIXED STATE"
        echo "🔄 Some containers may be starting/stopping"
    fi
}

show_cost_info() {
    echo "💰 Cost Breakdown & Optimization:"
    echo ""
    echo "🏃 RUNNING COSTS (per day):"
    echo "   Container Instances: ~$1.40/day"
    echo "   Storage: ~$0.15/day"
    echo "   Total: ~$1.55/day (~$47/month)"
    echo ""
    echo "💤 STOPPED COSTS (per day):"
    echo "   Container Instances: $0.00/day"
    echo "   Storage: ~$0.15/day" 
    echo "   Total: ~$0.15/day (~$4.50/month)"
    echo ""
    echo "💡 COST OPTIMIZATION TIPS:"
    echo "   • Use '$0 stop' when not actively using"
    echo "   • Use '$0 start' only when needed"
    echo "   • Consider Container Apps for auto scale-to-zero"
    echo "   • Delete entire deployment when project is done"
    echo ""
    echo "📊 SAVINGS POTENTIAL:"
    echo "   • Run 8 hours/day: ~$12/month (74% savings)"
    echo "   • Run 4 hours/day: ~$6/month (87% savings)"
    echo "   • Run only on weekends: ~$3/month (94% savings)"
}

case "$1" in
    start)
        start_containers
        ;;
    stop)
        stop_containers
        ;;
    status)
        show_status
        ;;
    cost)
        show_cost_info
        ;;
    *)
        show_help
        exit 1
        ;;
esac
