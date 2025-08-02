/**
 * Clean, scalable Results Display for ML Playground
 * Handles visualization of training results with flexible chart layout
 */
class ResultsDisplay {
    constructor(container) {
        console.log('🔧 ResultsDisplay constructor called with:', container, typeof container);
        
        // Handle both string ID and DOM element
        if (typeof container === 'string') {
            console.log('🔄 Converting string ID to DOM element:', container);
            const element = document.getElementById(container);
            if (!element) {
                console.error('❌ No element found with ID:', container);
                throw new Error(`No element found with ID: ${container}`);
            }
            this.container = element;
        } else if (container && container.nodeType === 1) {
            console.log('✅ Using provided DOM element');
            this.container = container;
        } else {
            console.error('❌ Invalid container provided:', container);
            throw new Error('Container must be a string ID or DOM element');
        }
        
        this.initializeContainer();
        console.log('✅ ResultsDisplay initialized with valid container');
    }

    /**
     * Initialize the main container structure
     */
    initializeContainer() {
        this.container.innerHTML = `
            <div class="results-container">
                <div class="results-sections">
                    <section class="sklearn-comparison-section">
                        <h3>📊 Performance Comparison</h3>
                        <div id="comparison-content">
                            <div class="loading">Calculating metrics...</div>
                        </div>
                    </section>
                    
                    <section class="visualizations-section">
                        <h3>📈 Visualizations</h3>
                        <div id="charts-grid" class="charts-grid">
                            <div class="loading">Generating charts...</div>
                        </div>
                    </section>
                    
                    <section class="training-details-section">
                        <h3>🔧 Training Details</h3>
                        <div id="training-details">
                            <div class="loading">Loading details...</div>
                        </div>
                    </section>
                </div>
            </div>
        `;
    }

    /**
     * Main method to display training results
     */
    async displayResults(result) {
        console.log('📊 Results display called with:', result);
        
        try {
            // Process result through data contract validation
            const dataContract = new DataContractValidator();
            const processedResult = dataContract.processTrainingResult(result);
            console.log('✅ Data contract processing completed');

            // Check if training was successful
            if (!processedResult.success) {
                console.error('❌ Training failed:', processedResult.error);
                this.displayError(processedResult.error || 'Training failed');
                return;
            }

            console.log('✅ Result successful, displaying results');
            
            // Display components
            this.displayComparison(processedResult);
            await this.displayCharts(processedResult);
            this.displayTrainingDetails(processedResult);
            
        } catch (error) {
            console.error('❌ Error displaying results:', error);
            this.displayError(`Error displaying results: ${error.message}`);
        }
    }

    /**
     * Display performance comparison metrics
     */
    displayComparison(result) {
        const container = document.getElementById('comparison-content');
        const yourMetrics = result.your_implementation?.metrics;
        const sklearnMetrics = result.sklearn_implementation?.metrics;

        if (!yourMetrics || !sklearnMetrics) {
            container.innerHTML = '<div class="chart-empty">Comparison data not available</div>';
            return;
        }

        // Determine which metrics to show based on algorithm type
        let metrics, metricNames;
        if (result.algorithm_type === 'binary_classification' || result.algorithm_type === 'multiclass_classification') {
            metrics = ['accuracy', 'precision', 'recall', 'f1_score'];
            metricNames = {
                accuracy: 'Accuracy',
                precision: 'Precision',
                recall: 'Recall',
                f1_score: 'F1 Score'
            };
        } else {
            // Default to regression metrics
            metrics = ['r2_score', 'mse', 'mae', 'rmse'];
            metricNames = {
                r2_score: 'R² Score',
                mse: 'Mean Squared Error',
                mae: 'Mean Absolute Error',
                rmse: 'Root Mean Squared Error'
            };
        }

        const metricsHTML = metrics.map(metric => {
            const yourValue = yourMetrics[metric];
            const sklearnValue = sklearnMetrics[metric];
            
            // More robust check for valid numeric values
            if (yourValue == null || sklearnValue == null || 
                typeof yourValue !== 'number' || typeof sklearnValue !== 'number' || 
                isNaN(yourValue) || isNaN(sklearnValue)) {
                return '';
            }

            const difference = yourValue - sklearnValue;
            
            // Determine if higher is better based on metric type
            let isBetter;
            if (result.algorithm_type === 'binary_classification' || result.algorithm_type === 'multiclass_classification') {
                // For classification metrics, higher is generally better
                isBetter = difference > 0;
            } else {
                // For regression metrics, R² higher is better, others lower is better
                const isR2 = metric === 'r2_score';
                isBetter = isR2 ? difference > 0 : difference < 0;
            }
            
            return `
                <div class="metric-card">
                    <div class="metric-name">${metricNames[metric]}</div>
                    <div class="metric-values">
                        <span class="metric-value my-impl">Mine: ${(yourValue || 0).toFixed(4)}</span>
                        <span class="metric-value sklearn-impl">Sklearn: ${(sklearnValue || 0).toFixed(4)}</span>
                        <span class="metric-value ${isBetter ? 'better' : 'worse'}">
                            ${difference > 0 ? '+' : ''}${(difference || 0).toFixed(4)}
                        </span>
                    </div>
                </div>
            `;
        }).filter(Boolean).join('');

        if (metricsHTML.trim() === '') {
            container.innerHTML = '<div class="chart-empty">No comparison metrics available for this algorithm type</div>';
        } else {
            container.innerHTML = `<div class="comparison-metrics">${metricsHTML}</div>`;
        }
    }

    /**
     * Display charts with flexible layout system
     */
    async displayCharts(result) {
        console.log('📊 Starting chart display...');
        const chartsGrid = document.getElementById('charts-grid');
        
        if (!result.charts) {
            chartsGrid.innerHTML = '<div class="chart-empty">No visualization data available</div>';
            return;
        }

        // Clear previous charts
        chartsGrid.innerHTML = '';
        const chartsToRender = [];

        // Detect available chart types and create chart configs
        const chartConfigs = this.detectAndConfigureCharts(result);
        
        // Create chart wrappers with appropriate sizing
        chartConfigs.forEach(config => {
            const wrapper = this.createFlexibleChartWrapper(config);
            chartsGrid.appendChild(wrapper);
            chartsToRender.push(config);
        });

        // Apply responsive grid layout
        this.applyResponsiveLayout(chartsGrid, chartConfigs.length);

        // Render charts with proper error handling
        for (const config of chartsToRender) {
            try {
                console.log(`📈 Rendering ${config.type} chart...`);
                await this.renderChart(config.type, config.data);
                console.log(`✅ Successfully rendered ${config.type} chart`);
            } catch (error) {
                console.error(`❌ Error rendering ${config.type} chart:`, error);
                this.displayChartError(config.id, `Error rendering ${config.type} chart: ${error.message}`);
            }
        }

        console.log(`📊 Successfully created ${chartsToRender.length} charts`);
    }

    /**
     * Detect available charts and configure them properly
     */
    detectAndConfigureCharts(result) {
        console.log('🔍 Chart detection starting with result.charts:', result.charts);
        const configs = [];

        // 1. Loss Curve - Training Progress
        if (result.charts.loss_curve && result.charts.loss_curve.iterations && result.charts.loss_curve.loss) {
            console.log('✅ Detected loss_curve chart');
            configs.push({
                type: 'loss_curve',
                data: result.charts.loss_curve,
                id: 'loss-curve-chart',
                title: 'Training Loss Convergence',
                size: 'medium',
                priority: 1
            });
        } else {
            console.log('❌ loss_curve not detected:', !!result.charts.loss_curve);
        }

        // 2. Predictions vs Actual - Model Performance
        if (result.charts.scatter_plot && result.charts.scatter_plot.actual) {
            console.log('✅ Detected scatter_plot chart');
            configs.push({
                type: 'predictions',
                data: result.charts.scatter_plot,
                id: 'predictions-chart',
                title: 'Predictions vs Actual Values',
                size: 'medium',
                priority: 2
            });
        } else {
            console.log('❌ scatter_plot not detected:', !!result.charts.scatter_plot);
        }

        // 3. Residuals Plot - Model Diagnostic
        if (result.charts.residuals_plot && result.charts.residuals_plot.your_residuals) {
            console.log('✅ Detected residuals_plot chart');
            configs.push({
                type: 'residuals',
                data: result.charts.residuals_plot,
                id: 'residuals-chart',
                title: 'Residual Analysis',
                size: 'medium',
                priority: 3
            });
        } else {
            console.log('❌ residuals_plot not detected:', !!result.charts.residuals_plot, 'has your_residuals:', !!result.charts.residuals_plot?.your_residuals);
        }

        // 4. Confusion Matrix - Classification Performance
        if (result.charts.confusion_matrix && result.charts.confusion_matrix.your_matrix) {
            configs.push({
                type: 'confusion_matrix',
                data: result.charts.confusion_matrix,
                id: 'confusion-matrix-chart',
                title: 'Confusion Matrix Comparison',
                size: 'large',
                priority: 1
            });
        }

        // 5. Model Coefficients - Feature Importance
        if (result.your_implementation?.coefficients) {
            configs.push({
                type: 'coefficients',
                data: result.your_implementation,
                id: 'coefficients-chart',
                title: 'Feature Importance (Coefficients)',
                size: 'medium',
                priority: 4
            });
        }

        // Sort by priority
        console.log(`🎯 Final chart configs detected: ${configs.length}`, configs.map(c => c.title));
        return configs.sort((a, b) => a.priority - b.priority);
    }

    /**
     * Create flexible chart wrapper with proper sizing
     */
    createFlexibleChartWrapper(config) {
        const wrapper = document.createElement('div');
        wrapper.className = `chart-wrapper chart-${config.size}`;
        wrapper.setAttribute('data-chart-type', config.type);
        
        wrapper.innerHTML = `
            <div class="chart-header">
                <h4 class="chart-title">${config.title}</h4>
                <div class="chart-controls">
                    <button class="chart-resize-btn" title="Toggle size">⤢</button>
                </div>
            </div>
            <div id="${config.id}" class="chart-container">
                <div class="chart-loading">
                    <div class="loading-spinner"></div>
                    <span>Preparing chart...</span>
                </div>
            </div>
        `;

        // Add resize functionality
        const resizeBtn = wrapper.querySelector('.chart-resize-btn');
        resizeBtn.addEventListener('click', () => this.toggleChartSize(wrapper));

        return wrapper;
    }

    /**
     * Apply responsive grid layout based on number of charts
     */
    applyResponsiveLayout(container, chartCount) {
        // Remove existing layout classes
        container.className = container.className.replace(/layout-\d+/g, '');
        
        // Apply appropriate layout class
        if (chartCount === 1) {
            container.classList.add('layout-single');
        } else if (chartCount === 2) {
            container.classList.add('layout-dual');
        } else if (chartCount <= 4) {
            container.classList.add('layout-grid');
        } else {
            container.classList.add('layout-masonry');
        }
    }

    /**
     * Toggle chart size between medium and large
     */
    toggleChartSize(wrapper) {
        if (wrapper.classList.contains('chart-large')) {
            wrapper.classList.remove('chart-large');
            wrapper.classList.add('chart-medium');
        } else {
            wrapper.classList.remove('chart-medium');
            wrapper.classList.add('chart-large');
        }
        
        // Trigger chart resize
        const chartId = wrapper.querySelector('.chart-container').id;
        setTimeout(() => {
            if (window.Plotly) {
                window.Plotly.Plots.resize(chartId);
            }
        }, 100);
    }

    /**
     * Main chart rendering dispatcher
     */
    async renderChart(type, data) {
        switch (type) {
            case 'loss_curve':
                return this.renderLossCurveChart(data);
            case 'predictions':
                return this.renderPredictionsChart(data);
            case 'residuals':
                return this.renderResidualsChart(data);
            case 'confusion_matrix':
                return this.renderConfusionMatrixChart(data);
            case 'coefficients':
                return this.renderCoefficientsChart(data);
            default:
                throw new Error(`Unknown chart type: ${type}`);
        }
    }

    /**
     * Render loss curve chart
     */
    async renderLossCurveChart(lossData) {
        const trace = {
            x: lossData.iterations || [],
            y: lossData.loss || [],
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#2563eb', width: 2 },
            marker: { size: 4, color: '#1d4ed8' },
            name: 'Training Loss'
        };

        const layout = {
            title: { text: lossData.title || 'Training Loss Curve', font: { size: 14 } },
            xaxis: { title: lossData.x_label || 'Iteration', titlefont: { size: 12 } },
            yaxis: { title: lossData.y_label || 'Loss', titlefont: { size: 12 } },
            margin: { l: 50, r: 30, t: 40, b: 50 },
            font: { size: 11 },
            showlegend: false,
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)'
        };

        return this.plotChart('loss-curve-chart', [trace], layout, 'line');
    }

    /**
     * Render predictions vs actual scatter plot
     */
    async renderPredictionsChart(scatterData) {
        const traces = [];
        
        // Your predictions
        traces.push({
            x: scatterData.actual || [],
            y: scatterData.your_predictions || [],
            mode: 'markers',
            type: 'scatter',
            name: 'Your Model',
            marker: { color: '#2563eb', size: 6, opacity: 0.7 }
        });

        // Sklearn predictions
        if (scatterData.sklearn_predictions) {
            traces.push({
                x: scatterData.actual || [],
                y: scatterData.sklearn_predictions,
                mode: 'markers',
                type: 'scatter',
                name: 'Sklearn Model',
                marker: { color: '#dc2626', size: 6, opacity: 0.7 }
            });
        }

        // Perfect prediction line
        if (scatterData.actual && scatterData.actual.length > 0) {
            const minVal = Math.min(...scatterData.actual);
            const maxVal = Math.max(...scatterData.actual);
            traces.push({
                x: [minVal, maxVal],
                y: [minVal, maxVal],
                mode: 'lines',
                type: 'scatter',
                name: 'Perfect Prediction',
                line: { color: '#6b7280', dash: 'dash', width: 1 },
                showlegend: false
            });
        }

        const layout = {
            title: { text: scatterData.title || 'Predictions vs Actual', font: { size: 14 } },
            xaxis: { title: scatterData.x_label || 'Actual Values', titlefont: { size: 12 } },
            yaxis: { title: scatterData.y_label || 'Predicted Values', titlefont: { size: 12 } },
            margin: { l: 50, r: 30, t: 40, b: 50 },
            font: { size: 11 },
            showlegend: traces.length > 2,
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)'
        };

        return this.plotChart('predictions-chart', traces, layout, 'scatter');
    }

    /**
     * Render residuals analysis chart
     */
    async renderResidualsChart(residualsData) {
        const traces = [];

        // Your residuals
        traces.push({
            x: residualsData.your_predictions || [],
            y: residualsData.your_residuals || [],
            mode: 'markers',
            type: 'scatter',
            name: 'Your Model Residuals',
            marker: { color: '#2563eb', size: 6, opacity: 0.7 }
        });

        // Sklearn residuals
        if (residualsData.sklearn_predictions && residualsData.sklearn_residuals) {
            traces.push({
                x: residualsData.sklearn_predictions,
                y: residualsData.sklearn_residuals,
                mode: 'markers',
                type: 'scatter',
                name: 'Sklearn Model Residuals',
                marker: { color: '#dc2626', size: 6, opacity: 0.7 }
            });
        }

        // Zero line
        if (residualsData.your_predictions && residualsData.your_predictions.length > 0) {
            const minPred = Math.min(...residualsData.your_predictions);
            const maxPred = Math.max(...residualsData.your_predictions);
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
            title: { text: residualsData.title || 'Residuals Analysis', font: { size: 14 } },
            xaxis: { title: residualsData.x_label || 'Predicted Values', titlefont: { size: 12 } },
            yaxis: { title: residualsData.y_label || 'Residuals', titlefont: { size: 12 } },
            margin: { l: 50, r: 30, t: 40, b: 50 },
            font: { size: 11 },
            showlegend: traces.length > 2,
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)'
        };

        return this.plotChart('residuals-chart', traces, layout, 'scatter');
    }

    /**
     * Render enhanced confusion matrix chart
     */
    async renderConfusionMatrixChart(matrixData) {
        const yourMatrix = matrixData.your_matrix || [];
        const sklearnMatrix = matrixData.sklearn_matrix || [];
        const labels = matrixData.labels || [];
        
        // Create side-by-side heatmaps for comparison
        const yourTrace = {
            z: yourMatrix,
            x: labels,
            y: labels,
            type: 'heatmap',
            colorscale: 'Blues',
            showscale: true,
            text: yourMatrix.map(row => row.map(val => val.toString())),
            texttemplate: '%{text}',
            textfont: { color: 'white', size: 16, family: 'Arial Black' },
            hoverongaps: false,
            hovertemplate: 'Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>',
            name: 'Your Implementation'
        };

        const sklearnTrace = {
            z: sklearnMatrix,
            x: labels,
            y: labels,
            type: 'heatmap',
            colorscale: 'Greens',
            showscale: true,
            text: sklearnMatrix.map(row => row.map(val => val.toString())),
            texttemplate: '%{text}',
            textfont: { color: 'white', size: 16, family: 'Arial Black' },
            hoverongaps: false,
            hovertemplate: 'Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>',
            xaxis: 'x2',
            yaxis: 'y2',
            name: 'Sklearn Implementation'
        };

        const layout = {
            title: { 
                text: 'Confusion Matrix Comparison', 
                font: { size: 16 },
                x: 0.5,
                xanchor: 'center'
            },
            grid: { 
                rows: 1, 
                columns: 2, 
                pattern: 'independent',
                xgap: 0.2,
                ygap: 0.1
            },
            annotations: [
                {
                    text: '<b>Your Implementation</b>',
                    x: 0.2,
                    y: 1.15,
                    xref: 'paper',
                    yref: 'paper',
                    showarrow: false,
                    font: { size: 14, color: '#1f77b4' },
                    xanchor: 'center',
                    yanchor: 'bottom'
                },
                {
                    text: '<b>Sklearn Implementation</b>',
                    x: 0.8,
                    y: 1.15,
                    xref: 'paper',
                    yref: 'paper',
                    showarrow: false,
                    font: { size: 14, color: '#2ca02c' },
                    xanchor: 'center',
                    yanchor: 'bottom'
                }
            ],
            xaxis: { 
                title: 'Predicted Class', 
                side: 'bottom', 
                domain: [0, 0.4],
                titlefont: { size: 12 }
            },
            yaxis: { 
                title: 'Actual Class', 
                autorange: 'reversed', 
                domain: [0, 1],
                titlefont: { size: 12 }
            },
            xaxis2: { 
                title: 'Predicted Class', 
                side: 'bottom', 
                domain: [0.6, 1],
                titlefont: { size: 12 }
            },
            yaxis2: { 
                title: 'Actual Class', 
                autorange: 'reversed', 
                domain: [0, 1],
                titlefont: { size: 12 }
            },
            margin: { l: 80, r: 30, t: 100, b: 70 },
            font: { size: 11 },
            showlegend: false,
            height: 400,
            width: null,
            autosize: false,
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)'
        };

        return this.plotChart('confusion-matrix-chart', [yourTrace, sklearnTrace], layout, 'confusion-matrix');
    }

    /**
     * Render coefficients bar chart
     */
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
            margin: { l: 50, r: 30, t: 40, b: 80 },
            font: { size: 11 },
            showlegend: false,
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)'
        };

        return this.plotChart('coefficients-chart', [trace], layout, 'bar');
    }

    /**
     * Standardize layout to prevent text overlap and ensure consistent spacing
     */
    standardizeChartLayout(layout, chartType = 'default') {
        const standardLayout = {
            ...layout,
            // Ensure consistent margins for all charts
            margin: {
                l: Math.max(layout.margin?.l || 80, 80),
                r: Math.max(layout.margin?.r || 50, 50),
                t: Math.max(layout.margin?.t || 100, 100),
                b: Math.max(layout.margin?.b || 80, 80),
                pad: 10
            },
            // Prevent text overlap with proper font sizing
            font: {
                family: 'Inter, Arial, sans-serif',
                size: Math.min(layout.font?.size || 12, 14),
                color: '#374151'
            },
            // Title positioning to prevent overlap
            title: layout.title ? {
                ...layout.title,
                x: 0.5,
                xanchor: 'center',
                y: layout.title.y || 0.95,
                yanchor: 'top',
                pad: { t: 20, b: 20 },
                font: {
                    size: Math.min(layout.title.font?.size || 16, 18),
                    color: '#1f2937'
                }
            } : undefined,
            // Standardize legend positioning
            legend: layout.legend ? {
                orientation: 'h',
                x: 0.5,
                xanchor: 'center',
                y: -0.15,
                yanchor: 'top',
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: '#e5e7eb',
                borderwidth: 1,
                font: { size: 11 },
                ...layout.legend
            } : undefined
        };

        // Chart-specific adjustments
        switch (chartType) {
            case 'confusion-matrix':
                standardLayout.margin.t = 120; // Extra space for dual titles
                standardLayout.height = 400;
                standardLayout.autosize = false;
                // Ensure annotations don't overlap
                if (standardLayout.annotations) {
                    standardLayout.annotations = standardLayout.annotations.map((ann, index) => ({
                        ...ann,
                        y: Math.max(ann.y || 1.12, 1.12),
                        xanchor: 'center',
                        yanchor: 'bottom',
                        font: {
                            ...ann.font,
                            size: Math.min(ann.font?.size || 14, 14)
                        }
                    }));
                }
                break;

            case 'comparison':
                standardLayout.margin.t = 100;
                standardLayout.margin.b = 100;
                break;

            case 'scatter':
            case 'line':
                standardLayout.margin.l = 90;
                standardLayout.margin.b = 90;
                standardLayout.height = 350;
                standardLayout.autosize = false;
                break;

            case 'bar':
                standardLayout.margin.b = 120; // Extra space for category labels
                standardLayout.height = 300;
                standardLayout.autosize = false;
                break;
        }

        // Ensure axis titles don't overlap with tick labels
        if (standardLayout.xaxis) {
            standardLayout.xaxis = {
                ...standardLayout.xaxis,
                titlefont: {
                    size: Math.min(standardLayout.xaxis.titlefont?.size || 12, 12),
                    ...standardLayout.xaxis.titlefont
                },
                tickfont: {
                    size: Math.min(standardLayout.xaxis.tickfont?.size || 10, 11),
                    ...standardLayout.xaxis.tickfont
                },
                title: standardLayout.xaxis.title ? {
                    standoff: 20,
                    ...standardLayout.xaxis.title
                } : undefined
            };
        }

        if (standardLayout.yaxis) {
            standardLayout.yaxis = {
                ...standardLayout.yaxis,
                titlefont: {
                    size: Math.min(standardLayout.yaxis.titlefont?.size || 12, 12),
                    ...standardLayout.yaxis.titlefont
                },
                tickfont: {
                    size: Math.min(standardLayout.yaxis.tickfont?.size || 10, 11),
                    ...standardLayout.yaxis.tickfont
                },
                title: standardLayout.yaxis.title ? {
                    standoff: 20,
                    ...standardLayout.yaxis.title
                } : undefined
            };
        }

        // Handle dual axis charts (like confusion matrix)
        if (standardLayout.xaxis2) {
            standardLayout.xaxis2 = {
                ...standardLayout.xaxis2,
                titlefont: {
                    size: Math.min(standardLayout.xaxis2.titlefont?.size || 12, 12),
                    ...standardLayout.xaxis2.titlefont
                },
                tickfont: {
                    size: Math.min(standardLayout.xaxis2.tickfont?.size || 10, 11),
                    ...standardLayout.xaxis2.tickfont
                }
            };
        }

        if (standardLayout.yaxis2) {
            standardLayout.yaxis2 = {
                ...standardLayout.yaxis2,
                titlefont: {
                    size: Math.min(standardLayout.yaxis2.titlefont?.size || 12, 12),
                    ...standardLayout.yaxis2.titlefont
                },
                tickfont: {
                    size: Math.min(standardLayout.yaxis2.tickfont?.size || 10, 11),
                    ...standardLayout.yaxis2.tickfont
                }
            };
        }

        return standardLayout;
    }

    /**
     * Enhanced chart plotting with responsive behavior
     */
    async plotChart(chartId, traces, layout, chartType = 'default') {
        try {
            const container = document.getElementById(chartId);
            if (!container) {
                throw new Error(`Chart container ${chartId} not found`);
            }

            // Clear loading state
            container.innerHTML = '';

            // Apply layout standardization to prevent text overlap
            const standardizedLayout = this.standardizeChartLayout(layout, chartType);

            // Enhanced layout with responsive config
            const responsiveLayout = {
                ...standardizedLayout,
                autosize: standardizedLayout.autosize !== false, // Only autosize if not explicitly disabled
                responsive: true
            };

            // Plot configuration
            const config = {
                responsive: true,
                displayModeBar: false,
                staticPlot: false,
                showTips: true
            };

            await window.Plotly.newPlot(chartId, traces, responsiveLayout, config);
            
            // Add resize observer for responsive behavior with height constraints
            if (window.ResizeObserver) {
                let isResizing = false;
                const resizeObserver = new ResizeObserver((entries) => {
                    if (isResizing) return;
                    
                    for (const entry of entries) {
                        const { width } = entry.contentRect;
                        if (width > 0) {
                            isResizing = true;
                            // Only resize width, keep height fixed
                            const update = {
                                width: width - 32, // Account for padding
                                height: responsiveLayout.height || 350 // Keep original height
                            };
                            window.Plotly.relayout(chartId, update).then(() => {
                                isResizing = false;
                            });
                        }
                    }
                });
                resizeObserver.observe(container);
            }

            console.log(`✅ Successfully plotted ${chartId}`);
        } catch (error) {
            console.error(`❌ Error plotting ${chartId}:`, error);
            this.displayChartError(chartId, error.message);
            throw error;
        }
    }

    /**
     * Display chart error with helpful message
     */
    displayChartError(chartId, message) {
        const container = document.getElementById(chartId);
        if (container) {
            container.innerHTML = `
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
     * Display training details
     */
    displayTrainingDetails(result) {
        const container = document.getElementById('training-details');
        
        const details = {
            'Algorithm': result.algorithm_id,
            'Dataset': result.dataset?.name || 'Unknown',
            'Training Duration': `${(result.total_duration || 0).toFixed(2)}s`,
            'Samples': result.dataset?.n_samples || 'Unknown',
            'Features': result.dataset?.n_features || 'Unknown',
            'Timestamp': result.timestamp ? new Date(result.timestamp).toLocaleString() : 'Unknown'
        };

        // Add hyperparameters
        if (result.hyperparameters) {
            Object.entries(result.hyperparameters).forEach(([key, value]) => {
                details[key] = value;
            });
        }

        const detailsHTML = Object.entries(details).map(([key, value]) => `
            <div class="info-item">
                <div class="info-label">${key}</div>
                <div class="info-value">${value}</div>
            </div>
        `).join('');

        container.innerHTML = `<div class="training-info">${detailsHTML}</div>`;
    }

    /**
     * Display error message
     */
    displayError(error) {
        this.container.innerHTML = `
            <div class="error-container">
                <div class="error-icon">❌</div>
                <h3>Training Error</h3>
                <p>${error}</p>
            </div>
        `;
    }
}

// Export for use in other files
window.ResultsDisplay = ResultsDisplay;
