/**
 * Metrics Display - Handles performance comparison metrics
 */
class MetricsDisplay {
    constructor() {
        this.metrics = new Map();
    }

    /**
     * Display performance comparison metrics
     */
    displayComparison(result) {
        const container = document.getElementById('comparison-content');
        if (!container) {
            console.error('❌ Comparison content container not found');
            return;
        }

        try {
            const metrics = this.extractMetrics(result);
            if (metrics.length === 0) {
                container.innerHTML = '<div class="no-metrics">No performance metrics available.</div>';
                return;
            }

            const metricsHTML = this.generateMetricsHTML(metrics);
            container.innerHTML = `<div class="comparison-metrics">${metricsHTML}</div>`;
            
            console.log('✅ Performance comparison displayed successfully');
        } catch (error) {
            console.error('❌ Error displaying comparison:', error);
            container.innerHTML = '<div class="comparison-error">Failed to display performance metrics</div>';
        }
    }

    /**
     * Extract metrics from result data
     */
    extractMetrics(result) {
        const metrics = [];
        
        // Extract metrics from the correct nested structure
        const yourMetrics = result.your_implementation?.metrics || result.metrics || {};
        const sklearnMetrics = result.sklearn_implementation?.metrics || result.sklearn_metrics || {};
        
        console.log('🔍 Extracting metrics:', {
            yourMetrics,
            sklearnMetrics,
            hasYourImpl: !!result.your_implementation,
            hasSklearnImpl: !!result.sklearn_implementation
        });
        
        // Get all unique metric keys from both implementations
        const allMetricKeys = new Set([
            ...Object.keys(yourMetrics),
            ...Object.keys(sklearnMetrics)
        ]);
        
        // Filter out unwanted metrics (convergence, iterations, etc.)
        const unwantedMetrics = ['convergence', 'iterations', 'converged', 'n_iters', 'max_iter', 'convergence_iterations'];
        
        allMetricKeys.forEach(key => {
            // Skip unwanted metrics (exact match or contains unwanted terms)
            if (unwantedMetrics.includes(key) || 
                key.toLowerCase().includes('convergence') || 
                key.toLowerCase().includes('iteration')) return;
            
            const yourValue = yourMetrics[key];
            const sklearnValue = sklearnMetrics[key];
            
            // Skip null/undefined values
            if (yourValue === null && sklearnValue === null) return;
            if (yourValue === undefined && sklearnValue === undefined) return;
            
            const metric = {
                name: this.formatMetricName(key),
                yourValue: yourValue,
                sklearnValue: sklearnValue,
                key: key
            };
                
            // Calculate difference and determine if better/worse
            if (metric.sklearnValue !== undefined && !isNaN(metric.yourValue) && !isNaN(metric.sklearnValue)) {
                metric.difference = metric.yourValue - metric.sklearnValue;
                metric.isBetter = this.isMetricBetter(key, metric.yourValue, metric.sklearnValue);
            }
            
            metrics.push(metric);
        });
        
        return metrics;
    }

    /**
     * Format metric name for display
     */
    formatMetricName(key) {
        const nameMap = {
            'r2_score': 'R² Score',
            'mse': 'Mean Squared Error',
            'mae': 'Mean Absolute Error',
            'rmse': 'Root Mean Squared Error',
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_score': 'F1 Score'
        };
        
        return nameMap[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    /**
     * Determine if a metric value is better than another
     */
    isMetricBetter(metricKey, yourValue, sklearnValue) {
        // Higher is better for these metrics
        const higherIsBetter = ['r2_score', 'accuracy', 'precision', 'recall', 'f1_score'];
        // Lower is better for these metrics  
        const lowerIsBetter = ['mse', 'mae', 'rmse', 'loss'];
        
        if (higherIsBetter.includes(metricKey)) {
            return yourValue > sklearnValue;
        } else if (lowerIsBetter.includes(metricKey)) {
            return yourValue < sklearnValue;
        }
        
        // Default: assume lower is better for error metrics
        return yourValue < sklearnValue;
    }

    /**
     * Generate HTML for metrics display with vertical boxes and horizontal metrics
     */
    generateMetricsHTML(metrics) {
        if (metrics.length === 0) {
            return '<div class="no-metrics">No performance metrics available.</div>';
        }

        // Group metrics for horizontal display
        const metricsRow = metrics.map(metric => {
            const yourValue = this.formatValue(metric.yourValue);
            const sklearnValue = metric.sklearnValue !== undefined ? 
                this.formatValue(metric.sklearnValue) : 'N/A';
            
            let difference = 'N/A';
            let diffClass = '';
            if (metric.difference !== undefined) {
                difference = this.formatValue(metric.difference, true);
                diffClass = metric.isBetter ? 'positive' : 'negative';
            }

            return {
                name: metric.name,
                yourValue,
                sklearnValue,
                difference,
                diffClass
            };
        });

        // Create horizontal metrics layout with vertical comparison boxes
        return `
            <div class="metrics-comparison-container">
                <div class="metrics-headers">
                    <div class="metric-header-spacer"></div>
                    ${metricsRow.map(m => `<div class="metric-header">${m.name}</div>`).join('')}
                </div>
                
                <div class="implementation-box my-implementation">
                    <div class="implementation-label">
                        <span class="impl-icon">🚀</span>
                        <span class="impl-name">Your</span>
                    </div>
                    <div class="metrics-row">
                        ${metricsRow.map(m => `<div class="metric-value">${m.yourValue}</div>`).join('')}
                    </div>
                </div>

                <div class="implementation-box sklearn-implementation">
                    <div class="implementation-label">
                        <span class="impl-icon">🔬</span>
                        <span class="impl-name">Sklearn</span>
                    </div>
                    <div class="metrics-row">
                        ${metricsRow.map(m => `<div class="metric-value">${m.sklearnValue}</div>`).join('')}
                    </div>
                </div>

                <div class="implementation-box difference-box">
                    <div class="implementation-label">
                        <span class="impl-icon">📊</span>
                        <span class="impl-name">Diff</span>
                    </div>
                    <div class="metrics-row">
                        ${metricsRow.map(m => `<div class="metric-value ${m.diffClass}">${m.difference}</div>`).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Format numeric values for display
     */
    formatValue(value, showSign = false) {
        if (value === null || value === undefined || isNaN(value)) {
            return 'N/A';
        }
        
        const num = parseFloat(value);
        let formatted;
        
        if (Math.abs(num) >= 1000) {
            formatted = num.toFixed(0);
        } else if (Math.abs(num) >= 1) {
            formatted = num.toFixed(4);
        } else {
            formatted = num.toFixed(6);
        }
        
        // Remove trailing zeros
        formatted = formatted.replace(/\.?0+$/, '');
        
        if (showSign && num > 0) {
            formatted = '+' + formatted;
        }
        
        return formatted;
    }

    /**
     * Add custom metric
     */
    addCustomMetric(name, yourValue, sklearnValue = null) {
        const metric = {
            name: name,
            yourValue: yourValue,
            sklearnValue: sklearnValue,
            key: name.toLowerCase().replace(/\s+/g, '_')
        };
        
        if (sklearnValue !== null) {
            metric.difference = yourValue - sklearnValue;
            metric.isBetter = this.isMetricBetter(metric.key, yourValue, sklearnValue);
        }
        
        this.metrics.set(metric.key, metric);
        return metric;
    }

    /**
     * Clear all metrics
     */
    clearMetrics() {
        this.metrics.clear();
    }
}

// Export for use in other modules
window.MetricsDisplay = MetricsDisplay;
