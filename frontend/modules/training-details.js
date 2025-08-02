/**
 * Training Details - Handles display of training information and hyperparameters
 */
class TrainingDetails {
    constructor() {
        this.details = new Map();
    }

    /**
     * Display training details
     */
    displayTrainingDetails(result) {
        const container = document.getElementById('training-details');
        if (!container) {
            console.error('❌ Training details container not found');
            return;
        }

        try {
            const details = this.extractDetails(result);
            const detailsHTML = this.generateDetailsHTML(details);
            container.innerHTML = `<div class="training-details">${detailsHTML}</div>`;
            
            console.log('✅ Training details displayed successfully');
        } catch (error) {
            console.error('❌ Error displaying training details:', error);
            container.innerHTML = '<div class="details-error">Failed to display training details</div>';
        }
    }

    /**
     * Extract details from result data
     */
    extractDetails(result) {
        const details = {
            'Algorithm': result.algorithm_id || 'Unknown',
            'Dataset': result.dataset?.name || 'Unknown',
            'Duration': this.formatDuration(result.total_duration || 0),
            'Data': `${this.formatNumber(result.dataset?.n_samples) || 'N/A'} × ${this.formatNumber(result.dataset?.n_features) || 'N/A'}`
        };

        // Add most important hyperparameters only
        if (result.hyperparameters) {
            const importantParams = ['learning_rate', 'alpha', 'regularization', 'C', 'kernel'];
            Object.entries(result.hyperparameters).forEach(([key, value]) => {
                if (importantParams.includes(key)) {
                    details[this.formatParameterName(key)] = this.formatParameterValue(value);
                }
            });
        }

        // Add key training info
        if (result.training_info) {
            this.addTrainingInfo(details, result.training_info);
        }

        return details;
    }

    /**
     * Add training-specific information
     */
    addTrainingInfo(details, trainingInfo) {
        if (trainingInfo.iterations !== undefined) {
            details['Iterations'] = this.formatNumber(trainingInfo.iterations);
        }
        
        if (trainingInfo.convergence !== undefined) {
            details['Convergence'] = trainingInfo.convergence ? 'Yes' : 'No';
        }
        
        if (trainingInfo.final_loss !== undefined) {
            details['Final Loss'] = this.formatNumber(trainingInfo.final_loss, 6);
        }
        
        if (trainingInfo.learning_rate) {
            details['Learning Rate'] = this.formatNumber(trainingInfo.learning_rate, 6);
        }
    }

    /**
     * Format parameter names for display
     */
    formatParameterName(key) {
        const nameMap = {
            'alpha': 'Learning Rate (α)',
            'tolerance': 'Tolerance',
            'learning_rate': 'Learning Rate',
            'max_iter': 'Max Iterations',
            'regularization': 'Regularization',
            'n_iters': 'Iterations',
            'tol': 'Tolerance',
            'C': 'Regularization (C)',
            'gamma': 'Gamma',
            'kernel': 'Kernel',
            'degree': 'Polynomial Degree'
        };
        
        return nameMap[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    /**
     * Format parameter values for display
     */
    formatParameterValue(value) {
        if (value === null || value === undefined) {
            return 'N/A';
        }
        
        if (typeof value === 'boolean') {
            return value ? 'Yes' : 'No';
        }
        
        if (typeof value === 'string') {
            return value;
        }
        
        if (typeof value === 'number') {
            return this.formatNumber(value);
        }
        
        return String(value);
    }

    /**
     * Format duration in seconds
     */
    formatDuration(seconds) {
        if (!seconds || seconds < 0.001) {
            return '<0.001s';
        }
        
        if (seconds < 1) {
            return `${(seconds * 1000).toFixed(0)}ms`;
        }
        
        if (seconds < 60) {
            return `${seconds.toFixed(2)}s`;
        }
        
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}m ${remainingSeconds.toFixed(1)}s`;
    }

    /**
     * Format numbers for display
     */
    formatNumber(value, decimals = null) {
        if (value === null || value === undefined || isNaN(value)) {
            return 'N/A';
        }
        
        const num = parseFloat(value);
        
        // Auto-determine decimal places if not specified
        if (decimals === null) {
            if (Number.isInteger(num)) {
                return num.toLocaleString();
            } else if (Math.abs(num) >= 1) {
                decimals = 3;
            } else {
                decimals = 6;
            }
        }
        
        let formatted = num.toFixed(decimals);
        
        // Remove trailing zeros for non-integers
        if (decimals > 0) {
            formatted = formatted.replace(/\.?0+$/, '');
        }
        
        return formatted;
    }

    /**
     * Format timestamp for display
     */
    formatTimestamp(timestamp) {
        if (!timestamp) {
            return 'Unknown';
        }
        
        try {
            const date = new Date(timestamp);
            return date.toLocaleString();
        } catch (error) {
            console.warn('Invalid timestamp:', timestamp);
            return 'Invalid Date';
        }
    }

    /**
     * Generate HTML for details display
     */
    generateDetailsHTML(details) {
        return Object.entries(details).map(([key, value]) => `
            <div class="detail-item">
                <div class="detail-label">${key}</div>
                <div class="detail-value">${value}</div>
            </div>
        `).join('');
    }

    /**
     * Add custom detail
     */
    addCustomDetail(label, value) {
        this.details.set(label, value);
    }

    /**
     * Remove detail
     */
    removeDetail(label) {
        this.details.delete(label);
    }

    /**
     * Clear all details
     */
    clearDetails() {
        this.details.clear();
    }

    /**
     * Export details as JSON
     */
    exportDetails() {
        return Object.fromEntries(this.details);
    }
}

// Export for use in other modules
window.TrainingDetails = TrainingDetails;
