/**
 * Dashboard JavaScript functionality
 * Handles model status checking, training, and UI interactions
 */

// Global variables
let modelsStatus = {};
let trainingInProgress = false;

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

/**
 * Initialize dashboard functionality
 */
function initializeDashboard() {
    // Check initial models status
    checkModelsStatus();
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize any charts on the page
    initializeCharts();
}

/**
 * Set up event listeners for dashboard interactions
 */
function setupEventListeners() {
    // Model training button
    const trainButton = document.querySelector('[onclick="trainModels()"]');
    if (trainButton) {
        trainButton.addEventListener('click', function(e) {
            e.preventDefault();
            trainModels();
        });
    }
    
    // Refresh status button
    const refreshButton = document.querySelector('[onclick="checkModelsStatus()"]');
    if (refreshButton) {
        refreshButton.addEventListener('click', function(e) {
            e.preventDefault();
            checkModelsStatus();
        });
    }
}

/**
 * Check the status of all ML models
 */
function checkModelsStatus() {
    showLoadingState('status');
    
    fetch('/api/models/status')
        .then(response => response.json())
        .then(data => {
            hideLoadingState('status');
            
            if (data.success) {
                modelsStatus = data.models_status;
                updateStatusBadges(data.models_status);
                updateModelCards(data.models_status);
            } else {
                console.error('Error checking model status:', data.error);
                showErrorMessage('Failed to check model status: ' + data.error);
            }
        })
        .catch(error => {
            hideLoadingState('status');
            console.error('Error:', error);
            showErrorMessage('Network error while checking model status');
        });
}

/**
 * Update status badges for each model
 */
function updateStatusBadges(status) {
    const badges = {
        'packaging': document.getElementById('packaging-status'),
        'carbon_footprint': document.getElementById('carbon-status'),
        'product_recommendation': document.getElementById('product-status'),
        'esg_score': document.getElementById('esg-status')
    };

    Object.keys(status).forEach(model => {
        const badge = badges[model];
        if (badge) {
            if (status[model]) {
                badge.textContent = 'Ready';
                badge.className = 'badge bg-success';
            } else {
                badge.textContent = 'Not Trained';
                badge.className = 'badge bg-warning';
            }
        }
    });
}

/**
 * Update model cards with additional status information
 */
function updateModelCards(status) {
    Object.keys(status).forEach(model => {
        const card = document.querySelector(`[id*="${model}"]`)?.closest('.card');
        if (card) {
            if (status[model]) {
                card.classList.remove('border-warning');
                card.classList.add('border-success');
            } else {
                card.classList.remove('border-success');
                card.classList.add('border-warning');
            }
        }
    });
}

/**
 * Train all ML models
 */
function trainModels() {
    if (trainingInProgress) {
        showErrorMessage('Training is already in progress. Please wait...');
        return;
    }
    
    trainingInProgress = true;
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    loadingModal.show();
    
    // Update button state
    const trainButton = document.querySelector('[onclick="trainModels()"]');
    if (trainButton) {
        trainButton.disabled = true;
        trainButton.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Training...';
    }
    
    fetch('/api/models/train', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        loadingModal.hide();
        trainingInProgress = false;
        
        // Reset button state
        if (trainButton) {
            trainButton.disabled = false;
            trainButton.innerHTML = '<i class="fas fa-cog me-1"></i>Train All Models';
        }
        
        if (data.success) {
            displayTrainingResults(data.training_results);
            // Refresh status after training
            setTimeout(() => checkModelsStatus(), 1000);
        } else {
            showErrorMessage('Error training models: ' + data.error);
        }
    })
    .catch(error => {
        loadingModal.hide();
        trainingInProgress = false;
        
        // Reset button state
        if (trainButton) {
            trainButton.disabled = false;
            trainButton.innerHTML = '<i class="fas fa-cog me-1"></i>Train All Models';
        }
        
        console.error('Error:', error);
        showErrorMessage('Network error during model training');
    });
}

/**
 * Display training results in a modal
 */
function displayTrainingResults(results) {
    const modalBody = document.getElementById('trainingResults');
    if (!modalBody) return;
    
    let html = '<div class="row">';
    let successCount = 0;
    let totalCount = Object.keys(results).length;
    
    Object.keys(results).forEach(model => {
        const result = results[model];
        const success = result.success;
        
        if (success) successCount++;
        
        html += `
            <div class="col-md-6 mb-3">
                <div class="card ${success ? 'border-success' : 'border-danger'}">
                    <div class="card-header">
                        <h6 class="mb-0">${formatModelName(model)}</h6>
                    </div>
                    <div class="card-body">
                        <span class="badge ${success ? 'bg-success' : 'bg-danger'} mb-2">
                            ${success ? 'Success' : 'Failed'}
                        </span>
                        ${success ? getSuccessMetrics(result) : getErrorMessage(result)}
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    
    // Add overall summary
    html += `
        <div class="alert ${successCount === totalCount ? 'alert-success' : 'alert-warning'}">
            <h6>Training Summary</h6>
            <p class="mb-0">${successCount}/${totalCount} models trained successfully</p>
        </div>
    `;
    
    modalBody.innerHTML = html;
    
    const trainingModal = new bootstrap.Modal(document.getElementById('trainingModal'));
    trainingModal.show();
}

/**
 * Format model name for display
 */
function formatModelName(model) {
    return model.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Get success metrics HTML for training results
 */
function getSuccessMetrics(result) {
    let html = '<div class="small">';
    
    if (result.accuracy) {
        html += `<div><strong>Accuracy:</strong> ${(result.accuracy * 100).toFixed(2)}%</div>`;
    }
    
    if (result.r2_score) {
        html += `<div><strong>R² Score:</strong> ${result.r2_score.toFixed(4)}</div>`;
    }
    
    if (result.rmse) {
        html += `<div><strong>RMSE:</strong> ${result.rmse.toFixed(2)}</div>`;
    }
    
    if (result.mae) {
        html += `<div><strong>MAE:</strong> ${result.mae.toFixed(2)}</div>`;
    }
    
    html += '</div>';
    return html;
}

/**
 * Get error message HTML for training results
 */
function getErrorMessage(result) {
    return `<div class="text-danger small">${result.error || 'Unknown error occurred'}</div>`;
}

/**
 * Initialize charts on the dashboard
 */
function initializeCharts() {
    // Performance chart
    const performanceCanvas = document.getElementById('performanceChart');
    if (performanceCanvas) {
        initPerformanceChart();
    }
}

/**
 * Initialize the performance chart
 */
function initPerformanceChart() {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;
    
    // Sample performance data - in a real app, this would come from the API
    const performanceData = {
        labels: ['Packaging', 'Carbon Footprint', 'Product Rec.', 'ESG Score'],
        datasets: [{
            label: 'Model Performance (%)',
            data: [85, 78, 82, 77],
            backgroundColor: [
                'rgba(13, 110, 253, 0.8)',
                'rgba(255, 193, 7, 0.8)',
                'rgba(25, 135, 84, 0.8)',
                'rgba(13, 202, 240, 0.8)'
            ],
            borderColor: [
                'rgba(13, 110, 253, 1)',
                'rgba(255, 193, 7, 1)',
                'rgba(25, 135, 84, 1)',
                'rgba(13, 202, 240, 1)'
            ],
            borderWidth: 1
        }]
    };
    
    new Chart(ctx, {
        type: 'bar',
        data: performanceData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y + '%';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Show loading state for specific operations
 */
function showLoadingState(operation) {
    const elements = {
        'status': document.querySelector('[onclick="checkModelsStatus()"]'),
        'training': document.querySelector('[onclick="trainModels()"]')
    };
    
    const element = elements[operation];
    if (element) {
        element.disabled = true;
        element.classList.add('loading');
        
        if (operation === 'status') {
            element.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Checking...';
        } else if (operation === 'training') {
            element.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Training...';
        }
    }
}

/**
 * Hide loading state for specific operations
 */
function hideLoadingState(operation) {
    const elements = {
        'status': document.querySelector('[onclick="checkModelsStatus()"]'),
        'training': document.querySelector('[onclick="trainModels()"]')
    };
    
    const element = elements[operation];
    if (element) {
        element.disabled = false;
        element.classList.remove('loading');
        
        if (operation === 'status') {
            element.innerHTML = '<i class="fas fa-refresh me-1"></i>Refresh Status';
        } else if (operation === 'training') {
            element.innerHTML = '<i class="fas fa-cog me-1"></i>Train All Models';
        }
    }
}

/**
 * Show error message to user
 */
function showErrorMessage(message) {
    // Create or update error alert
    let errorAlert = document.getElementById('dashboard-error-alert');
    
    if (!errorAlert) {
        errorAlert = document.createElement('div');
        errorAlert.id = 'dashboard-error-alert';
        errorAlert.className = 'alert alert-danger alert-dismissible fade show';
        errorAlert.innerHTML = `
            <strong>Error:</strong> <span id="error-message"></span>
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Insert at the top of the main content
        const mainContent = document.querySelector('main .container');
        if (mainContent) {
            mainContent.insertBefore(errorAlert, mainContent.firstChild);
        }
    }
    
    const messageSpan = document.getElementById('error-message');
    if (messageSpan) {
        messageSpan.textContent = message;
    }
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        if (errorAlert) {
            const alert = new bootstrap.Alert(errorAlert);
            alert.close();
        }
    }, 5000);
}

/**
 * Show success message to user
 */
function showSuccessMessage(message) {
    // Create or update success alert
    let successAlert = document.getElementById('dashboard-success-alert');
    
    if (!successAlert) {
        successAlert = document.createElement('div');
        successAlert.id = 'dashboard-success-alert';
        successAlert.className = 'alert alert-success alert-dismissible fade show';
        successAlert.innerHTML = `
            <strong>Success:</strong> <span id="success-message"></span>
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Insert at the top of the main content
        const mainContent = document.querySelector('main .container');
        if (mainContent) {
            mainContent.insertBefore(successAlert, mainContent.firstChild);
        }
    }
    
    const messageSpan = document.getElementById('success-message');
    if (messageSpan) {
        messageSpan.textContent = message;
    }
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        if (successAlert) {
            const alert = new bootstrap.Alert(successAlert);
            alert.close();
        }
    }, 3000);
}

/**
 * Utility function to format numbers
 */
function formatNumber(num, decimals = 2) {
    return Number(num).toFixed(decimals);
}

/**
 * Utility function to format percentages
 */
function formatPercentage(num, decimals = 1) {
    return (Number(num) * 100).toFixed(decimals) + '%';
}

/**
 * Debounce function for performance optimization
 */
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction() {
        const context = this;
        const args = arguments;
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(context, args);
    };
}

// Export functions for global access
window.checkModelsStatus = checkModelsStatus;
window.trainModels = trainModels;
window.showErrorMessage = showErrorMessage;
window.showSuccessMessage = showSuccessMessage;
