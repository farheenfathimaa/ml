from flask import Flask, render_template, request, jsonify
from ml_model import get_predictor
import os
import logging
import json
import time
from datetime import datetime
import socket

app = Flask(__name__)

# Configure logging for ELK Stack
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create ELK-compatible logger
elk_logger = logging.getLogger('ml_api')
handler = logging.FileHandler('/app/logs/ml_api.log')
handler.setFormatter(logging.Formatter(
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "service": "ml_api", '
    '"host": "' + socket.gethostname() + '", "message": %(message)s}'
))
elk_logger.addHandler(handler)
elk_logger.setLevel(logging.INFO)

def log_prediction(query_type, input_data, result, processing_time, client_ip):
    """Log prediction for ELK monitoring"""
    log_data = {
        "event_type": "ml_prediction",
        "query_type": query_type,
        "input_data": input_data,
        "processing_time_ms": round(processing_time * 1000, 2),
        "client_ip": client_ip,
        "timestamp": datetime.utcnow().isoformat(),
        "success": result.get('success', False)
    }
    
    if result.get('success'):
        if query_type == 'single':
            log_data.update({
                "predicted_category": result['result']['Predicted Cat Name'],
                "category_id": result['result']['Predicted Cat ID'],
                "confidence_score": result['result']['Confidence Score'],
                "original_query": result['result']['original_query']
            })
        else:
            log_data.update({
                "num_predictions": len(result.get('results', [])),
                "avg_confidence": sum(r['Confidence Score'] for r in result.get('results', [])) / len(result.get('results', [])) if result.get('results') else 0
            })
    else:
        log_data["error"] = result.get('error', 'Unknown error')
    
    elk_logger.info(json.dumps(log_data))

@app.route('/')
def home():
    elk_logger.info(json.dumps({"event_type": "page_access", "page": "home", "client_ip": request.remote_addr}))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    client_ip = request.remote_addr
    
    try:
        data = request.json
        query_type = data.get('type', 'single')
        
        elk_logger.info(json.dumps({
            "event_type": "prediction_request",
            "query_type": query_type,
            "client_ip": client_ip,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        predictor = get_predictor()
        
        if query_type == 'single':
            description = data.get('description', '').strip()
            if not description:
                result = {'error': 'Please provide a product description', 'success': False}
                log_prediction(query_type, description, result, time.time() - start_time, client_ip)
                return jsonify(result), 400
            
            prediction = predictor.predict_category(description)
            result = {
                'success': True,
                'type': 'single',
                'result': prediction
            }
            
        elif query_type == 'multiple':
            descriptions = data.get('descriptions', [])
            if not descriptions:
                result = {'error': 'Please provide product descriptions', 'success': False}
                log_prediction(query_type, descriptions, result, time.time() - start_time, client_ip)
                return jsonify(result), 400
            
            # Filter empty descriptions
            descriptions = [desc.strip() for desc in descriptions if desc.strip()]
            if not descriptions:
                result = {'error': 'Please provide valid product descriptions', 'success': False}
                log_prediction(query_type, descriptions, result, time.time() - start_time, client_ip)
                return jsonify(result), 400
            
            predictions = predictor.predict_multiple(descriptions)
            result = {
                'success': True,
                'type': 'multiple',
                'results': predictions
            }
        
        else:
            result = {'error': 'Invalid query type', 'success': False}
            log_prediction(query_type, data, result, time.time() - start_time, client_ip)
            return jsonify(result), 400
        
        # Log successful prediction
        log_prediction(query_type, data, result, time.time() - start_time, client_ip)
        return jsonify(result)
    
    except Exception as e:
        result = {'error': f'Prediction failed: {str(e)}', 'success': False}
        log_prediction(query_type if 'query_type' in locals() else 'unknown', 
                      data if 'data' in locals() else {}, 
                      result, time.time() - start_time, client_ip)
        elk_logger.error(json.dumps({
            "event_type": "prediction_error",
            "error": str(e),
            "client_ip": client_ip,
            "timestamp": datetime.utcnow().isoformat()
        }))
        return jsonify(result), 500

@app.route('/health')
def health():
    try:
        # Test predictor availability
        predictor = get_predictor()
        test_result = predictor.predict_category("test motor")
        
        health_data = {
            "status": "healthy",
            "message": "ML API is running",
            "timestamp": datetime.utcnow().isoformat(),
            "predictor_status": "operational" if test_result else "warning"
        }
        
        elk_logger.info(json.dumps({
            "event_type": "health_check",
            "status": "healthy",
            "client_ip": request.remote_addr
        }))
        
        return jsonify(health_data)
    except Exception as e:
        elk_logger.error(json.dumps({
            "event_type": "health_check_failed",
            "error": str(e),
            "client_ip": request.remote_addr
        }))
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route('/metrics')
def metrics():
    """Basic metrics endpoint for monitoring"""
    try:
        # You can expand this with actual metrics
        metrics_data = {
            "service": "ml_api",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "host": socket.gethostname()
        }
        return jsonify(metrics_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create logs directory
    os.makedirs('/app/logs', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
