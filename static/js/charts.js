/**
 * Chart utilities and configurations for Sustainability Analytics Platform
 * Provides reusable chart components and styling
 */

// Chart.js default configuration
Chart.defaults.responsive = true;
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.plugins.legend.display = true;

// Color palette for consistent theming
const CHART_COLORS = {
    primary: 'rgba(13, 110, 253, 0.8)',
    primaryBorder: 'rgba(13, 110, 253, 1)',
    secondary: 'rgba(108, 117, 125, 0.8)',
    secondaryBorder: 'rgba(108, 117, 125, 1)',
    success: 'rgba(25, 135, 84, 0.8)',
    successBorder: 'rgba(25, 135, 84, 1)',
    danger: 'rgba(220, 53, 69, 0.8)',
    dangerBorder: 'rgba(220, 53, 69, 1)',
    warning: 'rgba(255, 193, 7, 0.8)',
    warningBorder: 'rgba(255, 193, 7, 1)',
    info: 'rgba(13, 202, 240, 0.8)',
    infoBorder: 'rgba(13, 202, 240, 1)',
    light: 'rgba(248, 249, 250, 0.8)',
    lightBorder: 'rgba(248, 249, 250, 1)',
    dark: 'rgba(33, 37, 41, 0.8)',
    darkBorder: 'rgba(33, 37, 41, 1)'
};

// Multi-color palette for diverse datasets
const MULTI_COLORS = [
    CHART_COLORS.primary,
    CHART_COLORS.success,
    CHART_COLORS.warning,
    CHART_COLORS.danger,
    CHART_COLORS.info,
    CHART_COLORS.secondary,
    'rgba(111, 66, 193, 0.8)',
    'rgba(214, 51, 132, 0.8)',
    'rgba(253, 126, 20, 0.8)',
    'rgba(32, 201, 151, 0.8)'
];

const MULTI_COLORS_BORDER = [
    CHART_COLORS.primaryBorder,
    CHART_COLORS.successBorder,
    CHART_COLORS.warningBorder,
    CHART_COLORS.dangerBorder,
    CHART_COLORS.infoBorder,
    CHART_COLORS.secondaryBorder,
    'rgba(111, 66, 193, 1)',
    'rgba(214, 51, 132, 1)',
    'rgba(253, 126, 20, 1)',
    'rgba(32, 201, 151, 1)'
];

/**
 * Create a feature importance bar chart
 */
function createFeatureImportanceChart(canvasId, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    const features = Object.keys(data);
    const values = Object.values(data);
    
    const defaultOptions = {
        type: 'bar',
        data: {
            labels: features.map(f => f.length > 20 ? f.substring(0, 20) + '...' : f),
            datasets: [{
                label: 'Importance',
                data: values,
                backgroundColor: features.map((_, i) => MULTI_COLORS[i % MULTI_COLORS.length]),
                borderColor: features.map((_, i) => MULTI_COLORS_BORDER[i % MULTI_COLORS_BORDER.length]),
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.8)'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.8)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'rgba(255, 255, 255, 1)',
                    bodyColor: 'rgba(255, 255, 255, 0.8)',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1
                }
            }
        }
    };
    
    // Merge with custom options
    const chartOptions = { ...defaultOptions, ...options };
    
    return new Chart(ctx, chartOptions);
}

/**
 * Create a doughnut chart for categorical data
 */
function createDoughnutChart(canvasId, labels, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    const defaultOptions = {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: labels.map((_, i) => MULTI_COLORS[i % MULTI_COLORS.length]),
                borderColor: labels.map((_, i) => MULTI_COLORS_BORDER[i % MULTI_COLORS_BORDER.length]),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: 'rgba(255, 255, 255, 0.8)',
                        usePointStyle: true,
                        padding: 15
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'rgba(255, 255, 255, 1)',
                    bodyColor: 'rgba(255, 255, 255, 0.8)',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1
                }
            }
        }
    };
    
    // Merge with custom options
    const chartOptions = { ...defaultOptions, ...options };
    
    return new Chart(ctx, chartOptions);
}

/**
 * Create a radar chart for multi-dimensional data
 */
function createRadarChart(canvasId, labels, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    const defaultOptions = {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Values',
                data: data,
                backgroundColor: CHART_COLORS.success,
                borderColor: CHART_COLORS.successBorder,
                borderWidth: 2,
                pointBackgroundColor: CHART_COLORS.successBorder,
                pointBorderColor: CHART_COLORS.successBorder,
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    angleLines: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    pointLabels: {
                        color: 'rgba(255, 255, 255, 0.8)',
                        font: {
                            size: 11
                        }
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.6)',
                        backdropColor: 'transparent'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'rgba(255, 255, 255, 1)',
                    bodyColor: 'rgba(255, 255, 255, 0.8)',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1
                }
            }
        }
    };
    
    // Merge with custom options
    const chartOptions = { ...defaultOptions, ...options };
    
    return new Chart(ctx, chartOptions);
}

/**
 * Create a line chart for time series or continuous data
 */
function createLineChart(canvasId, labels, datasets, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    // Ensure datasets have proper styling
    const styledDatasets = datasets.map((dataset, index) => ({
        ...dataset,
        backgroundColor: dataset.backgroundColor || MULTI_COLORS[index % MULTI_COLORS.length],
        borderColor: dataset.borderColor || MULTI_COLORS_BORDER[index % MULTI_COLORS_BORDER.length],
        borderWidth: dataset.borderWidth || 2,
        fill: dataset.fill !== undefined ? dataset.fill : false,
        tension: dataset.tension || 0.1
    }));
    
    const defaultOptions = {
        type: 'line',
        data: {
            labels: labels,
            datasets: styledDatasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.8)'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.8)'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: 'rgba(255, 255, 255, 0.8)',
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'rgba(255, 255, 255, 1)',
                    bodyColor: 'rgba(255, 255, 255, 0.8)',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1
                }
            }
        }
    };
    
    // Merge with custom options
    const chartOptions = { ...defaultOptions, ...options };
    
    return new Chart(ctx, chartOptions);
}

/**
 * Create a bar chart for categorical data
 */
function createBarChart(canvasId, labels, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    const defaultOptions = {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Values',
                data: data,
                backgroundColor: labels.map((_, i) => MULTI_COLORS[i % MULTI_COLORS.length]),
                borderColor: labels.map((_, i) => MULTI_COLORS_BORDER[i % MULTI_COLORS_BORDER.length]),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.8)'
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.8)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'rgba(255, 255, 255, 1)',
                    bodyColor: 'rgba(255, 255, 255, 0.8)',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1
                }
            }
        }
    };
    
    // Merge with custom options
    const chartOptions = { ...defaultOptions, ...options };
    
    return new Chart(ctx, chartOptions);
}

/**
 * Update chart data dynamically
 */
function updateChartData(chart, newData, newLabels = null) {
    if (!chart) return;
    
    if (newLabels) {
        chart.data.labels = newLabels;
    }
    
    if (Array.isArray(newData)) {
        // Single dataset
        chart.data.datasets[0].data = newData;
    } else if (typeof newData === 'object') {
        // Multiple datasets
        Object.keys(newData).forEach((key, index) => {
            if (chart.data.datasets[index]) {
                chart.data.datasets[index].data = newData[key];
            }
        });
    }
    
    chart.update();
}

/**
 * Destroy chart safely
 */
function destroyChart(chart) {
    if (chart && typeof chart.destroy === 'function') {
        chart.destroy();
    }
}

/**
 * Get responsive chart height based on screen size
 */
function getResponsiveHeight() {
    const width = window.innerWidth;
    if (width < 576) return 200;
    if (width < 768) return 250;
    if (width < 992) return 300;
    return 350;
}

/**
 * Apply dark theme to chart options
 */
function applyDarkTheme(options) {
    if (!options.plugins) options.plugins = {};
    if (!options.scales) options.scales = {};
    
    // Apply dark theme colors to scales
    Object.keys(options.scales).forEach(scaleKey => {
        const scale = options.scales[scaleKey];
        if (!scale.grid) scale.grid = {};
        if (!scale.ticks) scale.ticks = {};
        
        scale.grid.color = 'rgba(255, 255, 255, 0.1)';
        scale.ticks.color = 'rgba(255, 255, 255, 0.8)';
    });
    
    // Apply dark theme to legend
    if (options.plugins.legend && options.plugins.legend.labels) {
        options.plugins.legend.labels.color = 'rgba(255, 255, 255, 0.8)';
    }
    
    // Apply dark theme to tooltip
    if (!options.plugins.tooltip) options.plugins.tooltip = {};
    options.plugins.tooltip = {
        ...options.plugins.tooltip,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'rgba(255, 255, 255, 1)',
        bodyColor: 'rgba(255, 255, 255, 0.8)',
        borderColor: 'rgba(255, 255, 255, 0.2)',
        borderWidth: 1
    };
    
    return options;
}

/**
 * Format chart values for display
 */
function formatChartValue(value, type = 'number') {
    switch (type) {
        case 'percentage':
            return (value * 100).toFixed(1) + '%';
        case 'currency':
            return '$' + value.toFixed(2);
        case 'decimal':
            return value.toFixed(2);
        case 'integer':
            return Math.round(value);
        default:
            return value;
    }
}

/**
 * Export chart as image
 */
function exportChartAsImage(chart, filename = 'chart.png') {
    if (!chart || !chart.canvas) return;
    
    const link = document.createElement('a');
    link.download = filename;
    link.href = chart.toBase64Image();
    link.click();
}

// Export functions for global access
window.ChartUtils = {
    CHART_COLORS,
    MULTI_COLORS,
    createFeatureImportanceChart,
    createDoughnutChart,
    createRadarChart,
    createLineChart,
    createBarChart,
    updateChartData,
    destroyChart,
    getResponsiveHeight,
    applyDarkTheme,
    formatChartValue,
    exportChartAsImage
};
