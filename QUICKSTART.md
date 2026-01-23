"""
Quick Start Guide for LATS API Server
"""

# LATS API Server - Quick Start

## Prerequisites

1. **Python 3.9+** installed
2. **DeepSeek API Key** (in `.env` file)
3. **Java Backend** running on `localhost:8080` (optional for testing)

## Installation

```bash
cd python-lats-server

# Install dependencies
pip install -r requirements.txt

# Verify .env file exists with API key
cat .env
```

## Running the Server

```bash
# Start the server
python main.py

# Server will start on http://localhost:8000
# API documentation: http://localhost:8000/docs
```

## Testing the API

### Option 1: Run Test Suite (No Java Backend Needed)

```bash
# In a new terminal (while server is running)
python test_api.py
```

### Option 2: Use Swagger UI

1. Open browser: http://localhost:8000/docs
2. Try the `/api/v1/lats/health` endpoint
3. Click "Try it out" → "Execute"

### Option 3: Use curl

```bash
# Health check
curl http://localhost:8000/api/v1/lats/health

# List sessions
curl http://localhost:8000/api/v1/lats/sessions

# Search request (requires Java backend)
curl -X POST http://localhost:8000/api/v1/lats/search \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_123",
    "function_signature": "int calculate(int x, int y)",
    "function_path": "src/Calculator.cpp::calculate",
    "function_code": "int calculate(int x, int y) { if (x > 0 && y < 100) { return x + y; } return 0; }",
    "context": "",
    "coverage_target": 0.95,
    "max_iterations": 20
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/api/v1/lats/health` | GET | Health check |
| `/api/v1/lats/search` | POST | Generate test suite |
| `/api/v1/lats/session/{id}` | GET | Get session info |
| `/api/v1/lats/session/{id}` | DELETE | Delete session |
| `/api/v1/lats/sessions` | GET | List all sessions |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc UI |

## Example Request/Response

### Request
```json
{
  "session_id": "session_123",
  "function_signature": "int calculate(int x, int y)",
  "function_path": "src/Calculator.cpp::calculate",
  "function_code": "int calculate(int x, int y) {\n  if (x > 0 && y < 100) {\n    return x + y;\n  }\n  return 0;\n}",
  "context": "",
  "coverage_target": 0.95,
  "max_iterations": 20
}
```

### Response
```json
{
  "session_id": "session_123",
  "status": "success",
  "test_names": ["test_001", "test_002", "test_003"],
  "final_coverage": 0.96,
  "iterations": 12,
  "total_tests_generated": 8,
  "total_tests_in_suite": 3,
  "tokens_used": 4521,
  "search_time_seconds": 8.5,
  "learned_rules": ["Rule 1", "Rule 2"],
  "coverage_details": {
    "statement": 1.0,
    "branch": 1.0,
    "mcdc": 0.96
  }
}
```

## Troubleshooting

### Server won't start

```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000                  # Linux/Mac

# Kill process on port 8000 if needed
# Then restart server
```

### Import errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Connection to Java backend fails

1. Verify Java backend is running: `curl http://localhost:8080/api/test-execution/execute`
2. Check `JAVA_BACKEND_URL` in `.env`
3. Check firewall settings

### LLM errors

1. Verify DeepSeek API key in `.env`
2. Check token budget (default: 100k)
3. Check internet connection

## Development

### Auto-reload

The server runs with `reload=True` by default. Any code changes will automatically reload the server.

### Logging

Set log level in `.env`:
```
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

### Running Tests

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest --cov=core --cov=api tests/
```

## Next Steps

1. ✅ API is running
2. Test with Java backend integration
3. Deploy to production server
4. Integrate with Java GUI

---

**Documentation**: See `LATS_WORKFLOW_DETAILED.md` for complete workflow
**GitHub**: https://github.com/Ming191/LATS-Implementation-For-AKA-AI
