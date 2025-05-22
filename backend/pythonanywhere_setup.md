# PythonAnywhere Deployment Instructions

## 1. Sign Up for PythonAnywhere
1. Go to [PythonAnywhere](https://www.pythonanywhere.com/) and sign up for a free account

## 2. Set Up a Web App
1. After logging in, go to the Web tab
2. Click "Add a new web app"
3. Choose "Manual configuration"
4. Select Python 3.11
5. Set the path to your web app (e.g., /home/yourusername/accent-detector)

## 3. Clone Your Repository
1. Go to the Consoles tab
2. Start a new Bash console
3. Clone your repository:
   ```bash
   git clone https://github.com/DlansiMalek/accent-detector.git
   ```

## 4. Set Up a Virtual Environment
1. In the Bash console, create a virtual environment:
   ```bash
   cd accent-detector
   python -m venv venv
   source venv/bin/activate
   pip install -r backend/requirements.txt
   ```

## 5. Configure the WSGI File
1. Go to the Web tab
2. Click on the WSGI configuration file link
3. Replace the content with:

```python
import sys
import os

# Add your project directory to the sys.path
path = '/home/yourusername/accent-detector/backend'
if path not in sys.path:
    sys.path.insert(0, path)

# Set environment variables
os.environ['PORT'] = '8000'

# Import your app from the backend directory
from main import app as application
```

## 6. Configure Static Files (Optional)
1. Go to the Web tab
2. Add a static files mapping if needed:
   - URL: /static/
   - Directory: /home/yourusername/accent-detector/backend/static

## 7. Configure CORS
1. Make sure your CORS settings in main.py include your frontend URL:
   ```python
   origins = [
       "https://accent-detector-app.netlify.app",
       "http://localhost:3000",
       "https://yourusername.pythonanywhere.com"
   ]
   ```

## 8. Reload the Web App
1. Go to the Web tab
2. Click the "Reload" button for your web app

Your backend should now be accessible at:
https://yourusername.pythonanywhere.com/

## 9. Update Frontend Configuration
1. Update your frontend's `.env.production` file to point to your new backend URL:
   ```
   REACT_APP_API_URL=https://yourusername.pythonanywhere.com
   ```
2. Redeploy your frontend to Netlify
