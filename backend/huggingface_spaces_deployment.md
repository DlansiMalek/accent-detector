# Hugging Face Spaces Deployment Instructions

Hugging Face Spaces is an excellent free option for deploying ML-based applications like your accent detector. It's specifically designed for machine learning workloads and offers generous resources on the free tier.

## Why Hugging Face Spaces?

1. **ML-Optimized**: Built specifically for machine learning applications
2. **Generous Free Tier**: 2GB RAM, 5GB storage
3. **Simple Deployment**: Direct GitHub integration
4. **Pre-installed ML Libraries**: Many ML dependencies are pre-installed
5. **No Credit Card Required**: Completely free to start

## Deployment Steps

### 1. Create a Hugging Face Account

1. Go to [Hugging Face](https://huggingface.co/) and sign up for a free account
2. Verify your email address

### 2. Create a New Space

1. Go to your profile and click "Create a Space"
2. Choose a name like "accent-detector-api"
3. Select "FastAPI" as the SDK
4. Choose "Public" visibility (or Private if you prefer)
5. Click "Create Space"

### 3. Connect Your GitHub Repository

1. In your Space, go to the "Settings" tab
2. Under "Repository", click "Connect to GitHub repository"
3. Select your "accent-detector" repository
4. Set the path to "/backend" since your FastAPI app is in the backend folder

### 4. Configure Environment Variables

1. In your Space's Settings, go to "Variables and Secrets"
2. Add any necessary environment variables (though none are required for your app)

### 5. Configure Space Hardware

1. In your Space's Settings, go to "Hardware"
2. Select "CPU" and "2GB RAM" (free tier)

### 6. Update CORS Settings

Make sure your CORS settings in `main.py` include your frontend URL:

```python
origins = [
    "https://accent-detector-app.netlify.app",
    "http://localhost:3000",
    "https://*.hf.space"  # Add this for Hugging Face Spaces
]
```

### 7. Update Frontend Configuration

Update your frontend's `.env.production` file to point to your new backend URL:

```
REACT_APP_API_URL=https://yourusername-accent-detector-api.hf.space
```

Replace `yourusername` with your actual Hugging Face username.

### 8. Redeploy Frontend to Netlify

Your frontend is already deployed to Netlify. You just need to update the backend URL in the `.env.production` file and redeploy.

## Monitoring and Management

- You can monitor your Space's logs directly in the Hugging Face interface
- Spaces automatically sleep after inactivity but wake up quickly when accessed
- You can restart your Space if needed from the Settings page

## Troubleshooting

If you encounter any issues:

1. Check the Space logs for error messages
2. Make sure your requirements.txt file includes all necessary dependencies
3. Verify that your main.py file is properly configured for the FastAPI app
4. Ensure your CORS settings allow requests from your frontend URL

Hugging Face Spaces is an ideal solution for ML applications like yours, as it's designed specifically for machine learning workloads and offers generous resources on the free tier.
