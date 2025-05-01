#!/bin/bash

set -e

VENV_DIR=".venv"
REQUIREMENTS_FILE="./requirements.txt"
MODEL_REPO_URL="https://github.com/manyakaistha/Phishing_Classifier_ML_Model.git"

echo_message() {
    echo "---- $1 ----"
}
echo_message "Checking for GIT"
if ! command -v git &> /dev/null; then
    echo "Error: git is required but not found. Please install Git." >&2
    exit 1
fi

echo_message "Clone ML Model Repository..."
if [ -d "Phishing_Classifier_ML_Model" ]; then
    echo "ML Model Repository already exists. Skipping cloning."
else
    echo "Cloning ML Model Repository..."
    git clone "$MODEL_REPO_URL"
    mv Phishing_Classifier_ML_Model trained_model
    echo "ML Model Repository cloned successfully."
fi

echo_message "Checking for virtual environment..."

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    echo "Creating virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
fi

echo_message "Installing dependencies from $REQUIREMENTS_FILE..."

if [ -f "$REQUIREMENTS_FILE" ]; then
    source "$VENV_DIR/bin/activate"
    pip install -r "$REQUIREMENTS_FILE"
    deactivate
    echo "Dependencies installed successfully."
else
    echo "Error: $REQUIREMENTS_FILE not found. Cannot install dependencies."
    exit 1
fi

echo_message "Setup complete!"
echo "You can now start the virtual environment by run'source $VENV_DIR/bin/activate'"
