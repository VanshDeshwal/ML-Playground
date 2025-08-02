/**
 * Chart Manager - Handles all chart visualization functionality
 */
class ChartManager {
    constructor() {
        this.chartConfigs = new Map();
        this.activeCharts = new Map();
    }

    /**
     * Detect and configure charts based on result data
     */
    detectAndConfigureCharts(result) {
        console.log('🔍 Detecting charts for result.charts:', result.charts);
        
        const charts = [];
        
        // Use the pre-built chart data from result.charts when available
        if (result.charts) {
            // 1. Loss Curve - Training Progress
            if (result.charts.loss_curve && result.charts.loss_curve.iterations && result.charts.loss_curve.loss) {
                console.log('✅ Detected loss_curve chart');
                charts.push({
                    id: 'loss-curve-chart',
                    title: 'Training Loss Convergence',
                    type: 'loss_curve',
                    size: 'medium',
                    data: result.charts.loss_curve,
                    priority: 1
                });
            }
            
            // 2. Scatter Plot - Predictions vs Actual
            if (result.charts.scatter_plot && result.charts.scatter_plot.actual) {
                console.log('✅ Detected scatter_plot chart');
                charts.push({
                    id: 'scatter-plot-chart',
                    title: `Predictions vs Actual - ${result.dataset?.name || 'Dataset'}`,
                    type: 'scatter_plot',
                    size: 'medium',
                    data: result.charts.scatter_plot,
                    priority: 2
                });
            }
            
            // 3. Residuals Plot - Model Diagnostic (for regression)
            if (result.charts.residuals_plot && result.charts.residuals_plot.your_residuals) {
                console.log('✅ Detected residuals_plot chart');
                charts.push({
                    id: 'residuals-chart',
                    title: 'Residual Analysis',
                    type: 'residuals_plot',
                    size: 'medium',
                    data: result.charts.residuals_plot,
                    priority: 3
                });
            }
            
            // 4. Confusion Matrix - Classification Performance
            if (result.charts.confusion_matrix && result.charts.confusion_matrix.your_matrix) {
                console.log('✅ Detected confusion_matrix chart');
                charts.push({
                    id: 'confusion-matrix-chart',
                    title: 'Confusion Matrix Comparison',
                    type: 'confusion_matrix',
                    size: 'large',
                    data: result.charts.confusion_matrix,
                    priority: 1
                });
            }
            
            // 5. ROC Curve - Classification Performance
            if (result.charts.roc_curve && result.charts.roc_curve.your_fpr) {
                console.log('✅ Detected roc_curve chart');
                charts.push({
                    id: 'roc-curve-chart',
                    title: 'ROC Curve Comparison',
                    type: 'roc_curve',
                    size: 'medium',
                    data: result.charts.roc_curve,
                    priority: 4
                });
            }
        }
        
        console.log(`📊 Detected ${charts.length} charts:`, charts.map(c => c.title));
        return charts;
    }

    /**
     * Create flexible chart wrapper with responsive design
     */
    createFlexibleChartWrapper(config) {
        const wrapper = document.createElement('div');
        wrapper.className = `chart-wrapper chart-${config.size}`;
        wrapper.setAttribute('data-chart-id', config.id);
        wrapper.setAttribute('data-priority', config.priority);
        
        wrapper.innerHTML = `
            <div class="chart-title">${config.title}</div>
            <div id="${config.id}" class="chart-container">
                <div class="chart-loading">Preparing chart...</div>
            </div>
        `;
        
        return wrapper;
    }

    /**
     * Apply responsive layout based on chart count
     */
    applyResponsiveLayout(container, chartCount) {
        container.classList.remove('charts-1', 'charts-2', 'charts-3', 'charts-4-plus');
        
        if (chartCount === 1) {
            container.classList.add('charts-1');
        } else if (chartCount === 2) {
            container.classList.add('charts-2');
        } else if (chartCount === 3) {
            container.classList.add('charts-3');
        } else {
            container.classList.add('charts-4-plus');
        }
    }

    /**
     * Clear chart loading state and prepare container for Plotly
     */
    clearChartLoading(chartId) {
        const chartContainer = document.getElementById(chartId);
        if (chartContainer) {
            chartContainer.innerHTML = ''; // Clear the loading content
        }
        return chartContainer;
    }

    /**
     * Display chart error
     */
    displayChartError(chartId, message) {
        const chartContainer = document.getElementById(chartId);
        if (chartContainer) {
            chartContainer.innerHTML = `
                <div class="chart-error">
                    <div class="error-icon">⚠️</div>
                    <div class="error-message">
                        <h4>Chart Error</h4>
                        <p>${message}</p>
                    </div>
                </div>
            `;
        }
    }

    /**
     * Create loss curve chart
     */
    async createLossCurveChart(chartId, data) {
        try {
            console.log('📊 Creating loss curve chart with data:', data);
            
            // Clear loading state
            const chartContainer = this.clearChartLoading(chartId);
            if (!chartContainer) {
                console.error('Chart container not found:', chartId);
                return;
            }
            
            // Handle both old and new data structures
            let xData, yData, title, xLabel, yLabel;
            
            if (data.iterations && data.loss) {
                // New structure from result.charts.loss_curve
                xData = data.iterations;
                yData = data.loss;
                title = data.title || 'Training Loss Curve';
                xLabel = data.x_label || 'Iteration';
                yLabel = data.y_label || 'Loss (MSE)';
            } else if (Array.isArray(data)) {
                // Old structure - array of loss values
                xData = Array.from({ length: data.length }, (_, i) => i);
                yData = data;
                title = 'Training Loss Curve';
                xLabel = 'Iteration';
                yLabel = 'Loss (MSE)';
            } else {
                this.displayChartError(chartId, 'No loss curve data available');
                return;
            }

            const trace = {
                x: xData,
                y: yData,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#3b82f6', width: 2 },
                name: 'Training Loss'
            };

            const layout = {
                title: title,
                xaxis: { title: xLabel },
                yaxis: { title: yLabel },
                showlegend: false,
                margin: { l: 60, r: 30, t: 60, b: 60 },
                autosize: true,
                width: null,
                height: null
            };

            await Plotly.newPlot(chartId, [trace], layout, { 
                responsive: true,
                displayModeBar: false,
                autosize: true
            });
            console.log('✅ Loss curve chart created successfully');
        } catch (error) {
            console.error('❌ Error creating loss curve chart:', error);
            this.displayChartError(chartId, 'Failed to create loss curve chart');
        }
    }

    /**
     * Create predictions vs actual scatter plot
     */
    async createPredictionsChart(chartId, data) {
        try {
            console.log('📊 Creating predictions chart with data:', data);
            
            // Clear loading state
            const chartContainer = this.clearChartLoading(chartId);
            if (!chartContainer) {
                console.error('Chart container not found:', chartId);
                return;
            }
            
            // Handle both old and new data structures
            let actualValues, yourPredictions, sklearnPredictions, title, xLabel, yLabel;
            
            if (data.actual && data.your_predictions) {
                // New structure from result.charts.scatter_plot
                actualValues = data.actual;
                yourPredictions = data.your_predictions;
                sklearnPredictions = data.sklearn_predictions;
                title = data.title || 'Predictions vs Actual Values';
                xLabel = data.x_label || 'Actual Values';
                yLabel = data.y_label || 'Predicted Values';
            } else if (data.your_predictions || data.sklearn_predictions) {
                // Old structure
                yourPredictions = data.your_predictions;
                sklearnPredictions = data.sklearn_predictions;
                actualValues = data.actual;
                title = `Predictions Comparison - ${data.dataset_name || 'Dataset'}`;
                xLabel = data.algorithm_type?.includes('classification') ? 'Sample Index' : 'Actual Values';
                yLabel = 'Predicted Values';
            } else {
                this.displayChartError(chartId, 'No prediction data available');
                return;
            }

            const isClassification = data.algorithm_type?.includes('classification');
            let traces = [];
            
            // Your implementation trace
            if (yourPredictions && yourPredictions.length > 0) {
                const yourTrace = {
                    x: actualValues || Array.from({length: yourPredictions.length}, (_, i) => i),
                    y: yourPredictions,
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Your Implementation',
                    marker: { color: '#3b82f6', size: 6, opacity: 0.7 }
                };
                traces.push(yourTrace);
            }

            // Sklearn comparison if available
            if (sklearnPredictions && sklearnPredictions.length > 0) {
                const sklearnTrace = {
                    x: actualValues || Array.from({length: sklearnPredictions.length}, (_, i) => i),
                    y: sklearnPredictions,
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Sklearn Implementation',
                    marker: { color: '#f59e0b', size: 6, opacity: 0.7 }
                };
                traces.push(sklearnTrace);
            }

            // Add perfect prediction line for regression
            if (actualValues && actualValues.length > 0 && !isClassification) {
                const minVal = Math.min(...actualValues);
                const maxVal = Math.max(...actualValues);
                traces.push({
                    x: [minVal, maxVal],
                    y: [minVal, maxVal],
                    mode: 'lines',
                    type: 'scatter',
                    name: 'Perfect Prediction',
                    line: { color: '#dc2626', dash: 'dash', width: 2 },
                    showlegend: false
                });
            }

            if (traces.length === 0) {
                this.displayChartError(chartId, 'No prediction data available');
                return;
            }

            const layout = {
                title: title,
                xaxis: { title: xLabel },
                yaxis: { title: yLabel },
                legend: { x: 0.02, y: 0.98 },
                margin: { l: 60, r: 30, t: 60, b: 60 },
                autosize: true,
                width: null,
                height: null
            };

            await Plotly.newPlot(chartId, traces, layout, { 
                responsive: true,
                displayModeBar: false,
                autosize: true
            });
            console.log('✅ Predictions chart created successfully');
        } catch (error) {
            console.error('❌ Error creating predictions chart:', error);
            this.displayChartError(chartId, 'Failed to create predictions chart');
        }
    }

    /**
     * Create residual analysis chart
     */
    async createResidualsChart(chartId, data) {
        try {
            console.log('📊 Creating residuals chart with data:', data);
            
            // Clear loading state
            const chartContainer = this.clearChartLoading(chartId);
            if (!chartContainer) {
                console.error('Chart container not found:', chartId);
                return;
            }
            
            // Handle both old and new data structures
            let yourPredictions, yourResiduals, sklearnPredictions, sklearnResiduals;
            
            if (data.your_predictions && data.your_residuals) {
                // New structure from result.charts.residuals_plot
                yourPredictions = data.your_predictions;
                yourResiduals = data.your_residuals;
                sklearnPredictions = data.sklearn_predictions;
                sklearnResiduals = data.sklearn_residuals;
            } else if (data.predictions && data.actual) {
                // Old structure - calculate residuals
                yourPredictions = data.predictions;
                yourResiduals = data.actual.map((actual, i) => actual - data.predictions[i]);
            } else {
                this.displayChartError(chartId, 'No residuals data available');
                return;
            }

            const traces = [];

            // Your implementation residuals
            if (yourPredictions && yourResiduals) {
                traces.push({
                    x: yourPredictions,
                    y: yourResiduals,
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Your Implementation',
                    marker: { color: '#3b82f6', size: 5, opacity: 0.7 }
                });
            }

            // Sklearn implementation residuals
            if (sklearnPredictions && sklearnResiduals) {
                traces.push({
                    x: sklearnPredictions,
                    y: sklearnResiduals,
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Sklearn Implementation',
                    marker: { color: '#f59e0b', size: 5, opacity: 0.7 }
                });
            }

            // Add zero line
            const allPredictions = [...(yourPredictions || []), ...(sklearnPredictions || [])];
            if (allPredictions.length > 0) {
                const minPred = Math.min(...allPredictions);
                const maxPred = Math.max(...allPredictions);
                traces.push({
                    x: [minPred, maxPred],
                    y: [0, 0],
                    mode: 'lines',
                    type: 'scatter',
                    name: 'Zero Line',
                    line: { color: '#6b7280', dash: 'dash', width: 1 },
                    showlegend: false
                });
            }

            const layout = {
                title: data.title || 'Residual Analysis',
                xaxis: { title: data.x_label || 'Predicted Values' },
                yaxis: { title: data.y_label || 'Residuals' },
                legend: { x: 0.02, y: 0.98 },
                margin: { l: 60, r: 30, t: 60, b: 60 },
                autosize: true,
                width: null,
                height: null
            };

            await Plotly.newPlot(chartId, traces, layout, { 
                responsive: true,
                displayModeBar: false,
                autosize: true
            });
            console.log('✅ Residuals chart created successfully');
        } catch (error) {
            console.error('❌ Error creating residuals chart:', error);
            this.displayChartError(chartId, 'Failed to create residuals chart');
        }
    }

    /**
     * Render all charts for a result
     */
    async renderCharts(result) {
        const chartsGrid = document.getElementById('charts-grid');
        if (!chartsGrid) {
            console.error('❌ Charts grid container not found');
            return;
        }

        try {
            const chartConfigs = this.detectAndConfigureCharts(result);
            
            if (chartConfigs.length === 0) {
                chartsGrid.innerHTML = '<div class="no-charts">No charts available for this result.</div>';
                return;
            }

            // Clear existing content
            chartsGrid.innerHTML = '';
            
            // Apply responsive layout
            this.applyResponsiveLayout(chartsGrid, chartConfigs.length);
            
            // Create chart wrappers
            chartConfigs.forEach(config => {
                const wrapper = this.createFlexibleChartWrapper(config);
                chartsGrid.appendChild(wrapper);
            });

            // Render individual charts
            for (const config of chartConfigs) {
                switch (config.type) {
                    case 'line':
                    case 'loss_curve':
                        await this.createLossCurveChart(config.id, config.data);
                        break;
                    case 'scatter':
                    case 'scatter_plot':
                        await this.createPredictionsChart(config.id, config.data);
                        break;
                    case 'residuals':
                    case 'residuals_plot':
                        await this.createResidualsChart(config.id, config.data);
                        break;
                    case 'classification':
                    case 'confusion_matrix':
                        await this.createConfusionMatrixChart(config.id, config.data);
                        break;
                    case 'roc_curve':
                        await this.createROCCurveChart(config.id, config.data);
                        break;
                    default:
                        console.warn('⚠️ Unknown chart type:', config.type);
                }
            }
            
            console.log('✅ All charts rendered successfully');
        } catch (error) {
            console.error('❌ Error rendering charts:', error);
            chartsGrid.innerHTML = '<div class="chart-error">Failed to render charts</div>';
        }
    }

    /**
     * Create classification accuracy comparison chart
     */
    async createClassificationChart(chartId, data) {
        try {
            // Clear loading state
            const chartContainer = this.clearChartLoading(chartId);
            if (!chartContainer) {
                console.error('Chart container not found:', chartId);
                return;
            }
            
            const accuracyData = [];
            const colors = [];
            const labels = [];

            if (data.accuracy_your !== undefined && data.accuracy_your !== null) {
                accuracyData.push(data.accuracy_your * 100);
                colors.push('#3b82f6');
                labels.push('Your Implementation');
            }

            if (data.accuracy_sklearn !== undefined && data.accuracy_sklearn !== null) {
                accuracyData.push(data.accuracy_sklearn * 100);
                colors.push('#f59e0b');
                labels.push('Sklearn Implementation');
            }

            if (accuracyData.length === 0) {
                this.displayChartError(chartId, 'No accuracy data available');
                return;
            }

            const trace = {
                x: labels,
                y: accuracyData,
                type: 'bar',
                marker: {
                    color: colors,
                    opacity: 0.8
                },
                text: accuracyData.map(val => `${val.toFixed(2)}%`),
                textposition: 'outside'
            };

            const layout = {
                title: 'Classification Accuracy Comparison',
                xaxis: { title: 'Implementation' },
                yaxis: { 
                    title: 'Accuracy (%)',
                    range: [0, 100]
                },
                showlegend: false,
                margin: { l: 50, r: 30, t: 50, b: 50 }
            };

            await Plotly.newPlot(chartId, [trace], layout, { responsive: true });
            console.log('✅ Classification chart created successfully');
        } catch (error) {
            console.error('❌ Error creating classification chart:', error);
            this.displayChartError(chartId, 'Failed to create classification chart');
        }
    }

    /**
     * Create confusion matrix chart
     */
    async createConfusionMatrixChart(chartId, data) {
        try {
            console.log('📊 Creating confusion matrix chart with data:', data);
            
            // Clear loading state
            const chartContainer = this.clearChartLoading(chartId);
            if (!chartContainer) {
                console.error('Chart container not found:', chartId);
                return;
            }
            
            const yourMatrix = data.your_matrix;
            const sklearnMatrix = data.sklearn_matrix;
            const labels = data.labels || ['Class 0', 'Class 1'];
            
            if (!yourMatrix || !sklearnMatrix) {
                this.displayChartError(chartId, 'No confusion matrix data available');
                return;
            }

            // Create side-by-side heatmaps
            const yourTrace = {
                z: yourMatrix,
                x: labels.map(l => `Predicted ${l}`),
                y: labels.map(l => `Actual ${l}`),
                type: 'heatmap',
                colorscale: 'Blues',
                showscale: false,
                text: yourMatrix.map(row => row.map(val => val.toString())),
                texttemplate: '%{text}',
                textfont: { color: 'white', size: 14 },
                xaxis: 'x',
                yaxis: 'y'
            };

            const sklearnTrace = {
                z: sklearnMatrix,
                x: labels.map(l => `Predicted ${l}`),
                y: labels.map(l => `Actual ${l}`),
                type: 'heatmap',
                colorscale: 'Oranges',
                showscale: false,
                text: sklearnMatrix.map(row => row.map(val => val.toString())),
                texttemplate: '%{text}',
                textfont: { color: 'white', size: 14 },
                xaxis: 'x2',
                yaxis: 'y2'
            };

            const layout = {
                title: 'Confusion Matrix Comparison',
                grid: { rows: 1, columns: 2, pattern: 'independent' },
                xaxis: { 
                    domain: [0, 0.45],
                    title: 'Your Implementation'
                },
                yaxis: { 
                    domain: [0, 1],
                    autorange: 'reversed'
                },
                xaxis2: { 
                    domain: [0.55, 1],
                    title: 'Sklearn Implementation'
                },
                yaxis2: { 
                    domain: [0, 1],
                    autorange: 'reversed'
                },
                font: { size: 12 },
                margin: { l: 80, r: 80, t: 60, b: 60 },
                autosize: true,
                width: null,
                height: null
            };

            await Plotly.newPlot(chartId, [yourTrace, sklearnTrace], layout, { 
                responsive: true,
                displayModeBar: false,
                autosize: true
            });
            console.log('✅ Confusion matrix chart created successfully');
        } catch (error) {
            console.error('❌ Error creating confusion matrix chart:', error);
            this.displayChartError(chartId, 'Failed to create confusion matrix chart');
        }
    }

    /**
     * Create ROC curve chart
     */
    async createROCCurveChart(chartId, data) {
        try {
            console.log('📊 Creating ROC curve chart with data:', data);
            
            // Clear loading state
            const chartContainer = this.clearChartLoading(chartId);
            if (!chartContainer) {
                console.error('Chart container not found:', chartId);
                return;
            }
            
            const yourFPR = data.your_fpr;
            const yourTPR = data.your_tpr;
            const sklearnFPR = data.sklearn_fpr;
            const sklearnTPR = data.sklearn_tpr;
            
            if (!yourFPR || !yourTPR) {
                this.displayChartError(chartId, 'No ROC curve data available');
                return;
            }

            const traces = [];

            // Your implementation ROC curve
            traces.push({
                x: yourFPR,
                y: yourTPR,
                type: 'scatter',
                mode: 'lines',
                name: 'Your Implementation',
                line: { color: '#3b82f6', width: 2 }
            });

            // Sklearn implementation ROC curve
            if (sklearnFPR && sklearnTPR) {
                traces.push({
                    x: sklearnFPR,
                    y: sklearnTPR,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Sklearn Implementation',
                    line: { color: '#f59e0b', width: 2 }
                });
            }

            // Diagonal line (random classifier)
            traces.push({
                x: [0, 1],
                y: [0, 1],
                type: 'scatter',
                mode: 'lines',
                name: 'Random Classifier',
                line: { color: '#6b7280', width: 1, dash: 'dash' }
            });

            const layout = {
                title: 'ROC Curve Comparison',
                xaxis: { title: 'False Positive Rate' },
                yaxis: { title: 'True Positive Rate' },
                font: { size: 12 },
                margin: { l: 60, r: 60, t: 60, b: 60 },
                legend: { x: 0.6, y: 0.2 }
            };

            await Plotly.newPlot(chartId, traces, layout, { responsive: true });
            console.log('✅ ROC curve chart created successfully');
        } catch (error) {
            console.error('❌ Error creating ROC curve chart:', error);
            this.displayChartError(chartId, 'Failed to create ROC curve chart');
        }
    }
}

// Export for use in other modules
window.ChartManager = ChartManager;
