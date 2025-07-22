function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Activate selected button
    event.target.classList.add('active');
}

async function predictSingle() {
    const input = document.getElementById('single-input').value.trim();
    const resultDiv = document.getElementById('single-result');
    
    if (!input) {
        showError(resultDiv, 'Please enter a product description');
        return;
    }
    
    showLoading(resultDiv);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                type: 'single',
                description: input
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displaySingleResult(resultDiv, data.result);
        } else {
            showError(resultDiv, data.error || 'Prediction failed');
        }
    } catch (error) {
        showError(resultDiv, 'Network error: ' + error.message);
    }
}

async function predictMultiple() {
    const input = document.getElementById('multiple-input').value.trim();
    const resultDiv = document.getElementById('multiple-result');
    
    if (!input) {
        showError(resultDiv, 'Please enter product descriptions');
        return;
    }
    
    const descriptions = input.split('\n').filter(desc => desc.trim());
    
    if (descriptions.length === 0) {
        showError(resultDiv, 'Please enter valid product descriptions');
        return;
    }
    
    showLoading(resultDiv);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                type: 'multiple',
                descriptions: descriptions
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayMultipleResults(resultDiv, data.results);
        } else {
            showError(resultDiv, data.error || 'Prediction failed');
        }
    } catch (error) {
        showError(resultDiv, 'Network error: ' + error.message);
    }
}

function displaySingleResult(container, result) {
    const confidenceClass = getConfidenceClass(result['Confidence Score']);
    
    let html = `
        <div class="result-card ${confidenceClass}">
            <div class="result-header">📦 ${result.original_query}</div>
            
            ${result.spelling_suggestions && result.spelling_suggestions.length > 0 ? 
                `<div class="spelling-suggestions">
                    💡 Did you mean: ${result.spelling_suggestions.join(', ')}
                </div>` : ''
            }
            
            <div class="result-details">
                <div class="detail-item">
                    <strong>Category:</strong><br>
                    ${result['Predicted Cat Name']}
                </div>
                <div class="detail-item">
                    <strong>Category ID:</strong><br>
                    ${result['Predicted Cat ID']}
                </div>
                <div class="detail-item">
                    <strong>Confidence:</strong><br>
                    ${(result['Confidence Score'] * 100).toFixed(2)}%
                </div>
                <div class="detail-item">
                    <strong>Processed Query:</strong><br>
                    ${result.used_query}
                </div>
            </div>
            
            ${result['Confidence Score'] < 0.6 ? 
                '<div class="spelling-suggestions">⚠️ Low confidence prediction - might need review</div>' : ''
            }
        </div>
    `;
    
    container.innerHTML = html;
}

function displayMultipleResults(container, results) {
    let html = `<h3>📊 Prediction Results (${results.length} products)</h3>`;
    
    results.forEach((result, index) => {
        const confidenceClass = getConfidenceClass(result['Confidence Score']);
        
        html += `
            <div class="result-card ${confidenceClass}">
                <div class="result-header">${index + 1}. 📦 ${result.original_query}</div>
                
                ${result.spelling_suggestions && result.spelling_suggestions.length > 0 ? 
                    `<div class="spelling-suggestions">
                        💡 Did you mean: ${result.spelling_suggestions.join(', ')}
                    </div>` : ''
                }
                
                <div class="result-details">
                    <div class="detail-item">
                        <strong>Category:</strong><br>
                        ${result['Predicted Cat Name']}
                    </div>
                    <div class="detail-item">
                        <strong>ID:</strong> ${result['Predicted Cat ID']} | 
                        <strong>Confidence:</strong> ${(result['Confidence Score'] * 100).toFixed(2)}%
                    </div>
                </div>
                
                ${result['Confidence Score'] < 0.6 ? 
                    '<div class="spelling-suggestions">⚠️ Low confidence - needs review</div>' : ''
                }
            </div>
        `;
    });
    
    container.innerHTML = html;
}

function getConfidenceClass(score) {
    if (score >= 0.8) return 'confidence-high';
    if (score >= 0.6) return 'confidence-medium';
    return 'confidence-low';
}

function showLoading(container) {
    container.innerHTML = '<div class="loading">🔄 Analyzing products...</div>';
}

function showError(container, message) {
    container.innerHTML = `<div class="error">❌ ${message}</div>`;
}

// Add Enter key support
document.getElementById('single-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        predictSingle();
    }
});
