from flask import Flask, render_template, request, jsonify
from flask.json.provider import DefaultJSONProvider
from ml_model import get_predictor
import os
import logging
import json
import time
from datetime import datetime
import socket
import numpy as np

app = Flask(__name__)

# Configure Flask to handle numpy types in JSON serialization
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Set the custom JSON provider for Flask 2.2+
app.json = NumpyJSONProvider(app)

# Create logs directory in the current working directory or user's home
def get_log_directory():
    """Get appropriate log directory based on environment"""
    if os.path.exists('/app') and os.access('/app', os.W_OK):
        # Docker/container environment
        return '/app/logs'
    else:
        # Local development environment
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'logs')

LOG_DIR = get_log_directory()
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging for ELK Stack
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enhanced ELK-compatible JSON formatter
class ELKJsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "@timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "service": "product-predictor",
            "host": socket.gethostname(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Parse message if it's JSON, otherwise use as string
        try:
            message_data = json.loads(record.getMessage())
            log_data.update(message_data)
        except (json.JSONDecodeError, ValueError):
            log_data["message"] = record.getMessage()
        
        return json.dumps(log_data)

# Create multiple loggers for different log types
def setup_loggers():
    # Main API logger
    api_logger = logging.getLogger('ml_api')
    api_handler = logging.FileHandler(os.path.join(LOG_DIR, 'ml_api.log'))
    api_handler.setFormatter(ELKJsonFormatter())
    api_logger.addHandler(api_handler)
    api_logger.setLevel(logging.INFO)
    
    # Predictions logger
    prediction_logger = logging.getLogger('predictions')
    prediction_handler = logging.FileHandler(os.path.join(LOG_DIR, 'predictions.log'))
    prediction_handler.setFormatter(ELKJsonFormatter())
    prediction_logger.addHandler(prediction_handler)
    prediction_logger.setLevel(logging.INFO)
    
    # Error logger
    error_logger = logging.getLogger('errors')
    error_handler = logging.FileHandler(os.path.join(LOG_DIR, 'errors.log'))
    error_handler.setFormatter(ELKJsonFormatter())
    error_logger.addHandler(error_handler)
    error_logger.setLevel(logging.ERROR)
    
    return api_logger, prediction_logger, error_logger

# Initialize loggers
elk_logger, prediction_logger, error_logger = setup_loggers()

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def log_prediction(query_type, input_data, result, processing_time, client_ip):
    """Enhanced prediction logging for ELK monitoring"""
    log_data = {
        "event_type": "ml_prediction",
        "query_type": query_type,
        "processing_time_ms": round(processing_time * 1000, 2),
        "client_ip": client_ip,
        "success": result.get('success', False),
        "user_agent": request.headers.get('User-Agent', 'Unknown'),
        "request_id": f"req_{int(time.time() * 1000)}"
    }
    
    # Add input data summary (not full data for privacy/size)
    if query_type == 'single':
        log_data["input_summary"] = {
            "description_length": len(str(input_data.get('description', '')))
        }
    else:
        descriptions = input_data.get('descriptions', [])
        log_data["input_summary"] = {
            "num_descriptions": len(descriptions),
            "avg_description_length": sum(len(str(d)) for d in descriptions) / len(descriptions) if descriptions else 0
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
            results = result.get('results', [])
            if results:
                log_data.update({
                    "num_predictions": len(results),
                    "avg_confidence": sum(r['Confidence Score'] for r in results) / len(results),
                    "min_confidence": min(r['Confidence Score'] for r in results),
                    "max_confidence": max(r['Confidence Score'] for r in results)
                })
    else:
        log_data["error"] = result.get('error', 'Unknown error')
        # Log to error logger as well
        error_logger.error(json.dumps({
            "event_type": "prediction_error",
            "error": result.get('error', 'Unknown error'),
            "client_ip": client_ip,
            "processing_time_ms": round(processing_time * 1000, 2)
        }))
    
    # Convert numpy types before logging
    log_data = convert_numpy_types(log_data)
    prediction_logger.info(json.dumps(log_data))

# Request logging middleware
@app.before_request
def log_request_start():
    request.start_time = time.time()
    elk_logger.info(json.dumps({
        "event_type": "request_start",
        "method": request.method,
        "url": request.url,
        "endpoint": request.endpoint,
        "client_ip": request.remote_addr,
        "user_agent": request.headers.get('User-Agent', 'Unknown')
    }))

@app.after_request
def log_request_end(response):
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        elk_logger.info(json.dumps({
            "event_type": "request_end",
            "method": request.method,
            "endpoint": request.endpoint,
            "status_code": response.status_code,
            "client_ip": request.remote_addr,
            "duration_ms": round(duration * 1000, 2)
        }))
    return response

@app.route('/')
def home():
    elk_logger.info(json.dumps({
        "event_type": "page_access", 
        "page": "home", 
        "client_ip": request.remote_addr
    }))
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
            "client_ip": client_ip
        }))
        
        predictor = get_predictor()
        
        if query_type == 'single':
            description = data.get('description', '').strip()
            if not description:
                result = {'error': 'Please provide a product description', 'success': False}
                log_prediction(query_type, data, result, time.time() - start_time, client_ip)
                return jsonify(result), 400
            
            prediction = predictor.predict_category(description)
            # Convert numpy types in prediction
            prediction = convert_numpy_types(prediction)
            
            result = {
                'success': True,
                'type': 'single',
                'result': prediction
            }
            
        elif query_type == 'multiple':
            descriptions = data.get('descriptions', [])
            if not descriptions:
                result = {'error': 'Please provide product descriptions', 'success': False}
                log_prediction(query_type, data, result, time.time() - start_time, client_ip)
                return jsonify(result), 400
            
            # Filter empty descriptions
            descriptions = [desc.strip() for desc in descriptions if desc.strip()]
            if not descriptions:
                result = {'error': 'Please provide valid product descriptions', 'success': False}
                log_prediction(query_type, data, result, time.time() - start_time, client_ip)
                return jsonify(result), 400
            
            predictions = predictor.predict_multiple(descriptions)
            # Convert numpy types in predictions
            predictions = convert_numpy_types(predictions)
            
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
        error_logger.error(json.dumps({
            "event_type": "prediction_exception",
            "error": str(e),
            "client_ip": client_ip,
            "exception_type": type(e).__name__
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
            "predictor_status": "operational" if test_result else "warning",
            "log_directory": LOG_DIR,
            "version": "1.0.0",
            "uptime": time.time()
        }
        
        elk_logger.info(json.dumps({
            "event_type": "health_check",
            "status": "healthy",
            "client_ip": request.remote_addr
        }))
        
        return jsonify(health_data)
    except Exception as e:
        error_logger.error(json.dumps({
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
    """Enhanced metrics endpoint for monitoring"""
    try:
        # You can expand this with actual metrics
        metrics_data = {
            "service": "product-predictor",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "host": socket.gethostname(),
            "log_directory": LOG_DIR,
            "python_version": f"{socket.gethostname()}",
            "disk_usage": {
                "logs_dir": LOG_DIR,
                "total_space": "N/A",  # You can add actual disk usage here
                "free_space": "N/A"
            }
        }
        
        elk_logger.info(json.dumps({
            "event_type": "metrics_access",
            "client_ip": request.remote_addr
        }))
        
        return jsonify(metrics_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# New endpoint for ELK testing
@app.route('/test-logging')
def test_logging():
    """Endpoint to test different log levels and types"""
    elk_logger.info(json.dumps({
        "event_type": "test_info_log",
        "message": "This is a test info log",
        "client_ip": request.remote_addr
    }))
    
    prediction_logger.info(json.dumps({
        "event_type": "test_prediction_log",
        "message": "This is a test prediction log",
        "client_ip": request.remote_addr
    }))
    
    error_logger.error(json.dumps({
        "event_type": "test_error_log",
        "message": "This is a test error log",
        "client_ip": request.remote_addr
    }))
    
    return jsonify({
        "message": "Test logs generated successfully",
        "timestamp": datetime.utcnow().isoformat(),
        "logs_written_to": [
            os.path.join(LOG_DIR, 'ml_api.log'),
            os.path.join(LOG_DIR, 'predictions.log'),
            os.path.join(LOG_DIR, 'errors.log')
        ]
    })

if __name__ == '__main__':
    print(f"Logs will be written to: {LOG_DIR}")
    print(f"Log files: ml_api.log, predictions.log, errors.log")
    
    # Log application startup
    elk_logger.info(json.dumps({
        "event_type": "application_startup",
        "service": "product-predictor",
        "log_directory": LOG_DIR,
        "host": socket.gethostname()
    }))
    
    app.run(host='0.0.0.0', port=5000, debug=False)