# Google Cloud Run Deployment Instructions

Google Cloud Run offers a generous free tier with better memory limits (up to 2GB) that should work well for your ML-based accent detection application.

## Prerequisites
1. A Google Cloud account (you can sign up for free and get $300 in credits)
2. Google Cloud CLI installed (gcloud)

## Deployment Steps

### 1. Set Up Google Cloud Project

```bash
# Install Google Cloud CLI if you haven't already
# Visit: https://cloud.google.com/sdk/docs/install

# Initialize gcloud and create a new project
gcloud init
gcloud projects create accent-detector-app
gcloud config set project accent-detector-app

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
```

### 2. Build and Deploy the Container

```bash
# Navigate to your backend directory
cd /home/malek/Desktop/spp/accent-detector/backend

# Build the container image using Cloud Build
gcloud builds submit --tag gcr.io/accent-detector-app/accent-detector-api

# Deploy to Cloud Run with 1GB memory (free tier allows up to 2GB)
gcloud run deploy accent-detector-api \
  --image gcr.io/accent-detector-app/accent-detector-api \
  --platform managed \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 1 \
  --allow-unauthenticated \
  --region us-central1
```

### 3. Update CORS Settings

Make sure your CORS settings in `main.py` include your frontend URL:

```python
origins = [
    "https://accent-detector-app.netlify.app",
    "http://localhost:3000"
]
```

### 4. Update Frontend Configuration

Update your frontend's `.env.production` file to point to your new backend URL:

```
REACT_APP_API_URL=https://accent-detector-api-xxxxx-uc.a.run.app
```

Replace `xxxxx` with your actual Cloud Run URL that will be provided after deployment.

### 5. Redeploy Frontend to Netlify

```bash
# Navigate to your frontend directory
cd /home/malek/Desktop/spp/accent-detector/frontend

# Build the frontend
npm run build

# Deploy to Netlify (if you have Netlify CLI installed)
netlify deploy --prod
```

## Monitoring and Scaling

- You can monitor your application in the Google Cloud Console
- Free tier includes 2 million requests per month
- Automatic scaling based on traffic (scales to zero when not in use)
- Memory can be increased up to 2GB in the free tier if needed

## Alternative: Deploy to Hugging Face Spaces

If Google Cloud Run setup seems complex, Hugging Face Spaces offers a simpler alternative:

1. Create an account at [Hugging Face](https://huggingface.co/)
2. Create a new Space with the "Gradio" or "FastAPI" template
3. Upload your backend code
4. Configure the Space to use Python 3.11
5. Add your requirements.txt file
6. The Space will automatically deploy your FastAPI application

Hugging Face Spaces offers a free tier with enough resources for your application and is specifically designed for ML applications.
