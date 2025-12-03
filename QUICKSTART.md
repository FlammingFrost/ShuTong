# Quick Start Guide

## Running ShuTong with React Frontend

### Step 1: Install Dependencies

#### Backend (Python)
```bash
# Install Flask dependencies
uv pip install flask flask-cors

# Or use requirements file
uv pip install -r requirements-api.txt
```

#### Frontend (React)
```bash
cd frontend
npm install
```

### Step 2: Start the Backend API

From the project root:
```bash
uv run python api_server.py
```

You should see:
```
Starting ShuTong API server...
API will be available at http://localhost:8000
```

### Step 3: Start the React Frontend

In a new terminal, from the `frontend` directory:
```bash
npm start
```

The browser will automatically open to `http://localhost:3000`

### Step 4: Explore the App

1. **Overview Page** - View model performance comparisons
2. **Problem Generator** - Generate math problems with AI
3. **Agent Solver** - Watch the agent solve problems in real-time

## Common Issues

### Port Already in Use

If port 8000 or 3000 is in use:

**Backend:**
```bash
python api_server.py --port 8080
```

Then update `frontend/.env`:
```
REACT_APP_API_URL=http://localhost:8080
```

**Frontend:**
```bash
PORT=3001 npm start
```

### Missing Dependencies

**Python:**
```bash
uv pip install -r requirements.txt
uv pip install -r requirements-api.txt
```

**Node:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### API Connection Failed

1. Check backend is running on port 8000
2. Verify no CORS errors in browser console
3. Check `REACT_APP_API_URL` in `.env`

## Production Deployment

### Build Frontend
```bash
cd frontend
npm run build
```

### Serve Static Files

Option 1 - Use Flask to serve React:
```python
# Add to api_server.py
from flask import send_from_directory

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

app.static_folder = '../frontend/build'
```

Option 2 - Use Nginx:
```nginx
server {
    listen 80;
    
    location / {
        root /path/to/frontend/build;
        try_files $uri /index.html;
    }
    
    location /api {
        proxy_pass http://localhost:8000;
    }
}
```

## Next Steps

- Customize the UI in `frontend/src/`
- Add new API endpoints in `api_server.py`
- Integrate with the existing agent system
- Add authentication if needed
- Deploy to production

Enjoy using ShuTong! 🧮
