// Clean, scalable Enhanced UI for ML Playground results
class EnhancedResultsDisplay {
    constructor(container) {
        console.log('🔧 EnhancedResultsDisplay constructor called with:', container, typeof container);
        
        // Handle both string ID and DOM element
        if (typeof container === 'string') {
            console.log('🔄 Converting string ID to DOM element:', container);
            const element = document.getElementById(container);
            if (!element) {
                console.error('❌ No element found with ID:', container);
                throw new Error(`No element found with ID: ${container}`);
            }
            this.container = element;
        } else if (container && container.innerHTML !== undefined) {
            this.container = container;
        } else {
            console.error('❌ Invalid container passed to EnhancedResultsDisplay:', container);
            throw new Error('Container must be a DOM element or valid element ID');
        }
        
        this.plotlyLoaded = false;
        console.log('✅ EnhancedResultsDisplay initialized with valid container');
    }

    async displayResults(result) {
        console.log('📊 Enhanced results display called with:', result);
        
        if (!result || !result.success) {
            this.displayError(result?.error || 'Training failed');
            return;
        }

        console.log('✅ Result successful, displaying enhanced results');
        
        // Clear previous results
        this.container.innerHTML = '';

        // Create main results container with clean structure
        const resultsContainer = document.createElement('div');
        resultsContainer.className = 'enhanced-results-container';
        resultsContainer.innerHTML = `
            <div class="results-sections">
                <section class="sklearn-comparison-section">
                    <h3>📊 Performance Comparison</h3>
                    <div id="comparison-content"></div>
                </section>
                
                <section class="visualizations-section">
                    <h3>📈 Visualizations</h3>
                    <div id="charts-grid" class="charts-grid"></div>
                </section>
                
                <section class="training-details-section">
                    <h3>🔧 Training Details</h3>
                    <div id="training-details"></div>
                </section>
            </div>
        `;

        // Add responsive CSS
        this.addResponsiveStyles();
        this.container.appendChild(resultsContainer);

        // Display content in order
        this.displayComparison(result);
        await this.displayVisualizations(result);
        this.displayTrainingDetails(result);
    }

    addResponsiveStyles() {
        // Remove any existing styles
        const existingStyle = document.getElementById('enhanced-ui-styles');
        if (existingStyle) {
            existingStyle.remove();
        }

        const style = document.createElement('style');
        style.id = 'enhanced-ui-styles';
        style.textContent = `
            .enhanced-results-container {
                max-width: 100%;
                margin: 0 auto;
                padding: 16px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #f8fafc;
            }

            .results-sections {
                display: flex;
                flex-direction: column;
                gap: 32px;
                max-width: 1400px;
                margin: 0 auto;
            }

            .results-sections section {
                background: #ffffff;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                border: 1px solid #e2e8f0;
            }

            .results-sections h3 {
                margin: 0 0 16px 0;
                color: #1e293b;
                font-size: 1.1rem;
                font-weight: 600;
                border-bottom: 2px solid #f1f5f9;
                padding-bottom: 8px;
            }

            /* Charts Grid - Fully Responsive */
            .charts-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 20px;
                width: 100%;
            }

            /* Chart Wrapper */
            .chart-wrapper {
                background: #f8fafc;
                border-radius: 8px;
                padding: 16px;
                border: 1px solid #e2e8f0;
                width: 100%;
                box-sizing: border-box;
            }

            .chart-title {
                font-size: 0.95rem;
                font-weight: 600;
                color: #475569;
                margin-bottom: 12px;
                text-align: center;
            }

            .chart-container {
                width: 100%;
                height: 350px;
                background: white;
                border-radius: 6px;
                border: 1px solid #e2e8f0;
                position: relative;
                overflow: hidden;
            }

            /* Loading and Error States */
            .chart-loading, .chart-error, .chart-empty {
                display: flex;
                align-items: center;
                justify-content: center;
                height: 350px;
                color: #64748b;
                font-size: 0.9rem;
                background: white;
                border-radius: 6px;
                border: 1px solid #e2e8f0;
            }

            .chart-loading::before { content: "⏳ "; margin-right: 8px; }
            .chart-error { color: #ef4444; }
            .chart-error::before { content: "❌ "; margin-right: 8px; }
            .chart-empty::before { content: "📊 "; margin-right: 8px; }

            /* Responsive Breakpoints */
            @media (min-width: 640px) {
                .enhanced-results-container { padding: 24px; }
                .results-sections section { padding: 24px; }
                .chart-container { height: 400px; }
                .chart-loading, .chart-error, .chart-empty { height: 400px; }
            }

            @media (min-width: 768px) {
                .charts-grid { grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); }
            }

            @media (min-width: 1024px) {
                .enhanced-results-container { padding: 32px; }
                .results-sections section { padding: 32px; }
                .charts-grid { grid-template-columns: repeat(2, 1fr); }
            }

            @media (min-width: 1400px) {
                .charts-grid { grid-template-columns: repeat(3, 1fr); }
            }

            /* Comparison Metrics */
            .comparison-metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
            }

            .metric-card {
                background: #f8fafc;
                border-radius: 8px;
                padding: 16px;
                border: 1px solid #e2e8f0;
            }

            .metric-name {
                font-size: 0.875rem;
                color: #64748b;
                font-weight: 500;
                margin-bottom: 8px;
            }

            .metric-values {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 8px;
                flex-wrap: wrap;
            }

            .metric-value {
                font-size: 0.95rem;
                font-weight: 600;
                padding: 4px 8px;
                border-radius: 4px;
            }

            .my-impl { color: #3b82f6; background: #dbeafe; }
            .sklearn-impl { color: #f59e0b; background: #fef3c7; }
            .better { color: #10b981; background: #d1fae5; }
            .worse { color: #ef4444; background: #fee2e2; }

            /* Training Details */
            .training-info {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 16px;
            }

            .info-item {
                background: #f8fafc;
                border-radius: 8px;
                padding: 14px;
                border: 1px solid #e2e8f0;
            }

            .info-label {
                font-size: 0.8rem;
                color: #64748b;
                font-weight: 500;
                margin-bottom: 4px;
                text-transform: uppercase;
                letter-spacing: 0.025em;
            }

            .info-value {
                font-size: 0.95rem;
                color: #1e293b;
                font-weight: 600;
                word-break: break-all;
            }

            /* Mobile Optimizations */
            @media (max-width: 640px) {
                .enhanced-results-container { padding: 12px; }
                .results-sections { gap: 20px; }
                .results-sections section { padding: 16px; }
                .results-sections h3 { font-size: 1rem; }
                .chart-container { height: 300px; }
                .chart-loading, .chart-error, .chart-empty { height: 300px; font-size: 0.85rem; }
                .metric-values { flex-direction: column; align-items: flex-start; }
                .training-info { grid-template-columns: 1fr; }
            }
        `;
        document.head.appendChild(style);
    }

    async displayVisualizations(result) {
        console.log('📈 Starting visualization display with result:', result);
        const chartsGrid = document.getElementById('charts-grid');
        
        if (!result.charts) {
            console.log('❌ No charts data found in result');
            chartsGrid.innerHTML = '<div class="chart-empty">No visualization data available</div>';
            return;
        }

        console.log('📊 Charts data found:', result.charts);
        
        // Ensure Plotly is loaded
        if (!window.Plotly) {
            console.log('⏳ Loading Plotly...');
            chartsGrid.innerHTML = '<div class="chart-loading">Loading visualization library...</div>';
            await this.loadPlotly();
        }

        chartsGrid.innerHTML = '';
        const chartsToRender = [];

        // 1. Loss Curve - Training Progress (Check for iterations and loss arrays)
        if (result.charts.loss_curve && result.charts.loss_curve.iterations && result.charts.loss_curve.loss) {
            console.log('✅ Creating loss curve chart with iterations:', result.charts.loss_curve.iterations.length);
            chartsToRender.push({
                type: 'loss_curve',
                data: result.charts.loss_curve,
                wrapper: this.createChartWrapper('Training Loss Convergence', 'loss-curve-chart')
            });
        }

        // 2. Predictions vs Actual - Model Performance (Check for actual and predictions arrays)
        if (result.charts.scatter_plot && result.charts.scatter_plot.actual) {
            console.log('✅ Creating predictions vs actual chart with', result.charts.scatter_plot.actual.length, 'data points');
            chartsToRender.push({
                type: 'predictions',
                data: result.charts.scatter_plot,
                wrapper: this.createChartWrapper('Predictions vs Actual Values', 'predictions-chart')
            });
        }

        // 3. Residuals Plot - Model Diagnostic (Check for residuals data)
        if (result.charts.residuals_plot && result.charts.residuals_plot.your_residuals) {
            console.log('✅ Creating residuals chart with', result.charts.residuals_plot.your_residuals.length, 'residual points');
            chartsToRender.push({
                type: 'residuals',
                data: result.charts.residuals_plot,
                wrapper: this.createChartWrapper('Residual Analysis', 'residuals-chart')
            });
        }

        // 4. Model Coefficients - Feature Importance (if available)
        if (result.your_implementation?.coefficients) {
            console.log('✅ Creating coefficients chart with', result.your_implementation.coefficients.length, 'features');
            chartsToRender.push({
                type: 'coefficients',
                data: result.your_implementation,
                wrapper: this.createChartWrapper('Feature Importance (Coefficients)', 'coefficients-chart')
            });
        }

        // First, add all chart wrappers to the DOM
        chartsToRender.forEach(chart => {
            chartsGrid.appendChild(chart.wrapper);
        });

        // Then, render the actual charts (DOM elements now exist)
        for (const chart of chartsToRender) {
            try {
                await this.renderChart(chart.type, chart.data);
            } catch (error) {
                console.error(`❌ Failed to render ${chart.type} chart:`, error);
            }
        }

        // Display result message
        if (chartsToRender.length === 0) {
            chartsGrid.innerHTML = '<div class="chart-empty">No charts could be generated from the training data</div>';
        } else {
            console.log(`📊 Successfully created ${chartsToRender.length} charts`);
        }
    }

    async renderChart(type, data) {
        switch (type) {
            case 'loss_curve':
                return this.renderLossCurveChart(data);
            case 'predictions':
                return this.renderPredictionsChart(data);
            case 'residuals':
                return this.renderResidualsChart(data);
            case 'coefficients':
                return this.renderCoefficientsChart(data);
            default:
                throw new Error(`Unknown chart type: ${type}`);
        }
    }

    async renderLossCurveChart(lossData) {
        const trace = {
            x: lossData.iterations,
            y: lossData.loss,
            type: 'scatter',
            mode: 'lines',
            name: 'Training Loss',
            line: { color: '#3b82f6', width: 2 }
        };

        const layout = {
            title: { text: lossData.title || 'Training Loss Curve', font: { size: 14 } },
            xaxis: { title: lossData.x_label || 'Iteration', titlefont: { size: 12 } },
            yaxis: { title: lossData.y_label || 'Loss', titlefont: { size: 12 } },
            margin: { l: 50, r: 30, t: 30, b: 50 },
            font: { size: 11 },
            showlegend: false
        };

        return this.plotChart('loss-curve-chart', [trace], layout);
    }

    async renderPredictionsChart(scatterData) {
        const traces = [];

        // My implementation
        if (scatterData.your_predictions && scatterData.actual) {
            traces.push({
                x: scatterData.actual,
                y: scatterData.your_predictions,
                mode: 'markers',
                type: 'scatter',
                name: 'My Implementation',
                marker: { color: '#3b82f6', size: 6, opacity: 0.7 }
            });
        }

        // Sklearn comparison
        if (scatterData.sklearn_predictions && scatterData.actual) {
            traces.push({
                x: scatterData.actual,
                y: scatterData.sklearn_predictions,
                mode: 'markers',
                type: 'scatter',
                name: 'Sklearn',
                marker: { color: '#f59e0b', size: 6, opacity: 0.7 }
            });
        }

        // Perfect prediction line
        if (scatterData.actual?.length > 0) {
            const min = Math.min(...scatterData.actual);
            const max = Math.max(...scatterData.actual);
            traces.push({
                x: [min, max],
                y: [min, max],
                mode: 'lines',
                type: 'scatter',
                name: 'Perfect Prediction',
                line: { color: '#dc2626', dash: 'dash', width: 2 },
                showlegend: false
            });
        }

        const layout = {
            title: { text: scatterData.title || 'Predictions vs Actual', font: { size: 14 } },
            xaxis: { title: scatterData.x_label || 'Actual Values', titlefont: { size: 12 } },
            yaxis: { title: scatterData.y_label || 'Predicted Values', titlefont: { size: 12 } },
            margin: { l: 50, r: 30, t: 30, b: 50 },
            font: { size: 11 },
            showlegend: traces.length > 1
        };

        return this.plotChart('predictions-chart', traces, layout);
    }

    async renderResidualsChart(residualsData) {
        const traces = [];

        // My implementation residuals
        if (residualsData.your_residuals && residualsData.your_predictions) {
            traces.push({
                x: residualsData.your_predictions,
                y: residualsData.your_residuals,
                mode: 'markers',
                type: 'scatter',
                name: 'My Implementation',
                marker: { color: '#3b82f6', size: 5, opacity: 0.7 }
            });
        }

        // Zero line
        if (residualsData.your_predictions?.length > 0) {
            const min = Math.min(...residualsData.your_predictions);
            const max = Math.max(...residualsData.your_predictions);
            traces.push({
                x: [min, max],
                y: [0, 0],
                mode: 'lines',
                type: 'scatter',
                name: 'Zero Line',
                line: { color: '#6b7280', dash: 'dash', width: 1 },
                showlegend: false
            });
        }

        const layout = {
            title: { text: residualsData.title || 'Residuals Analysis', font: { size: 14 } },
            xaxis: { title: residualsData.x_label || 'Predicted Values', titlefont: { size: 12 } },
            yaxis: { title: residualsData.y_label || 'Residuals', titlefont: { size: 12 } },
            margin: { l: 50, r: 30, t: 30, b: 50 },
            font: { size: 11 },
            showlegend: traces.length > 2
        };

        return this.plotChart('residuals-chart', traces, layout);
    }

    async renderCoefficientsChart(implementation) {
        const coefficients = implementation.coefficients || [];
        const intercept = implementation.intercept || 0;
        
        const allCoeffs = [intercept, ...coefficients];
        const labels = ['Intercept', ...coefficients.map((_, i) => `Feature ${i+1}`)];
        
        const trace = {
            x: labels,
            y: allCoeffs,
            type: 'bar',
            marker: {
                color: allCoeffs.map(val => val >= 0 ? '#10b981' : '#ef4444'),
                line: { color: '#374151', width: 1 }
            }
        };

        const layout = {
            title: { text: 'Feature Coefficients', font: { size: 14 } },
            xaxis: { title: 'Parameters', titlefont: { size: 12 } },
            yaxis: { title: 'Coefficient Value', titlefont: { size: 12 } },
            margin: { l: 50, r: 30, t: 30, b: 80 },
            font: { size: 11 },
            showlegend: false
        };

        return this.plotChart('coefficients-chart', [trace], layout);
    }

    createChartWrapper(title, chartId) {
        const wrapper = document.createElement('div');
        wrapper.className = 'chart-wrapper';
        wrapper.innerHTML = `
            <div class="chart-title">${title}</div>
            <div id="${chartId}" class="chart-container">
                <div class="chart-loading">Preparing chart...</div>
            </div>
        `;
        return wrapper;
    }

    async plotChart(chartId, traces, layout) {
        try {
            await window.Plotly.newPlot(chartId, traces, layout, {
                responsive: true,
                displayModeBar: false,
                staticPlot: false
            });
            console.log(`✅ Successfully plotted ${chartId}`);
        } catch (error) {
            console.error(`❌ Error plotting ${chartId}:`, error);
            const chartContainer = document.getElementById(chartId);
            if (chartContainer) {
                chartContainer.innerHTML = '<div class="chart-error">Failed to render chart</div>';
            }
        }
    }

    async loadPlotly() {
        if (window.Plotly) {
            this.plotlyLoaded = true;
            return;
        }

        try {
            const script = document.createElement('script');
            script.src = 'https://cdn.plot.ly/plotly-2.32.0.min.js';
            
            await new Promise((resolve, reject) => {
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });

            this.plotlyLoaded = true;
            console.log('✅ Plotly loaded successfully');
        } catch (error) {
            console.error('❌ Failed to load Plotly:', error);
            throw error;
        }
    }

    displayComparison(result) {
        const container = document.getElementById('comparison-content');
        const yourMetrics = result.your_implementation?.metrics;
        const sklearnMetrics = result.sklearn_implementation?.metrics;

        if (!yourMetrics || !sklearnMetrics) {
            container.innerHTML = '<div class="chart-empty">Comparison data not available</div>';
            return;
        }

        const metrics = ['r2_score', 'mse', 'mae', 'rmse'];
        const metricNames = {
            r2_score: 'R² Score',
            mse: 'Mean Squared Error',
            mae: 'Mean Absolute Error',
            rmse: 'Root Mean Squared Error'
        };

        const metricsHTML = metrics.map(metric => {
            const yourValue = yourMetrics[metric];
            const sklearnValue = sklearnMetrics[metric];
            
            if (yourValue === undefined || sklearnValue === undefined) return '';

            const difference = yourValue - sklearnValue;
            const isR2 = metric === 'r2_score';
            const isBetter = isR2 ? difference > 0 : difference < 0;
            
            return `
                <div class="metric-card">
                    <div class="metric-name">${metricNames[metric]}</div>
                    <div class="metric-values">
                        <span class="metric-value my-impl">Mine: ${yourValue.toFixed(4)}</span>
                        <span class="metric-value sklearn-impl">Sklearn: ${sklearnValue.toFixed(4)}</span>
                        <span class="metric-value ${isBetter ? 'better' : 'worse'}">
                            ${difference > 0 ? '+' : ''}${difference.toFixed(4)}
                        </span>
                    </div>
                </div>
            `;
        }).filter(Boolean).join('');

        container.innerHTML = `<div class="comparison-metrics">${metricsHTML}</div>`;
    }

    displayTrainingDetails(result) {
        const container = document.getElementById('training-details');
        const details = [
            { label: 'Algorithm', value: result.algorithm_id || 'Unknown' },
            { label: 'Dataset', value: result.dataset?.name || 'Unknown' },
            { label: 'Training Duration', value: `${(result.total_duration || 0).toFixed(2)}s` },
            { label: 'Samples', value: result.dataset?.n_samples || 'Unknown' },
            { label: 'Features', value: result.dataset?.n_features || 'Unknown' },
            { label: 'Timestamp', value: new Date(result.timestamp || Date.now()).toLocaleString() }
        ];

        const hyperparams = result.hyperparameters || {};
        Object.entries(hyperparams).forEach(([key, value]) => {
            details.push({ label: key, value: String(value) });
        });

        const detailsHTML = details.map(({ label, value }) => `
            <div class="info-item">
                <div class="info-label">${label}</div>
                <div class="info-value">${value}</div>
            </div>
        `).join('');

        container.innerHTML = `<div class="training-info">${detailsHTML}</div>`;
    }

    displayError(error) {
        this.container.innerHTML = `
            <div style="text-align: center; padding: 40px; color: #ef4444;">
                <h3>❌ Training Failed</h3>
                <p>${error}</p>
            </div>
        `;
    }
}

// Export for use in other files
window.EnhancedResultsDisplay = EnhancedResultsDisplay;
