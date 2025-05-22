#!/bin/bash

# Deploy to Hugging Face Spaces
# This script creates a new git repository for the backend and pushes it to Hugging Face Spaces

# Set variables
SPACE_NAME="accent-detector-api"
HF_USERNAME="DlansiMalek"  # Replace with your Hugging Face username

# Create a temporary directory for deployment
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Copy backend files to temporary directory
cp -r * $TEMP_DIR/
cp .hf-space $TEMP_DIR/
cp .dockerignore $TEMP_DIR/

# Navigate to temporary directory
cd $TEMP_DIR

# Initialize git repository
git init
git config --local user.email "you@example.com"
git config --local user.name "Your Name"

# Add all files
git add .

# Commit changes
git commit -m "Initial deployment of accent detector API"

# Create Hugging Face Space using the Hugging Face CLI
# Note: You need to have the Hugging Face CLI installed and be logged in
# To install: pip install huggingface_hub
# To login: huggingface-cli login
echo "Creating Hugging Face Space: $HF_USERNAME/$SPACE_NAME"
echo "You will need to have the Hugging Face CLI installed and be logged in"
echo "To install: pip install huggingface_hub"
echo "To login: huggingface-cli login"

# Create the space if it doesn't exist
huggingface-cli repo create $SPACE_NAME --type space --organization $HF_USERNAME --exist-ok

# Add remote and push
git remote add origin https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME
git push -u origin main --force

echo "Deployment complete!"
echo "Your API will be available at: https://$HF_USERNAME-$SPACE_NAME.hf.space"
echo "It may take a few minutes for the space to build and start"
