#!/bin/bash

# Azure ML Playground Cleanup Script
# This script removes all Azure resources created for ML Playground

set -e

# Configuration
RESOURCE_GROUP="ml-playground-rg"

echo "🧹 Starting ML Playground cleanup..."

# Confirm deletion
read -p "Are you sure you want to delete the resource group '$RESOURCE_GROUP' and all its resources? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled"
    exit 1
fi

echo "🗑️  Deleting resource group and all resources..."
az group delete --name $RESOURCE_GROUP --yes --no-wait

echo "✅ Cleanup initiated. Resources are being deleted in the background."
echo "💡 You can check the status with: az group show --name $RESOURCE_GROUP"
