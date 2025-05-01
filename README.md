# Phishing Email Classifier

A machine learning-based CLI application that detects phishing emails using DistilBERT and LSTM neural networks.

## Features
- CLI interface with rich visual feedback
- Pre-trained model included
- Model taining sript included

## Prerequisites
- Python 3
- pip

## Installation
Clone this repository:
```bash
git clone
cd phishing-email-classifier
```

### Automatically using setup.sh
This appraoch will install all the required dependencies and set up the virtual environment for you using python virtual environment.
1. Clone this repository (Already done if you are folled the previous steps)
2. Navigate to the project directory (Already done if you are folled the previous steps)
3. Make the `setup.sh` executable:
    ```bash
    chmod +x setup.sh
    ```
4. Run the setup script:
    ```bash
    ./setup.sh
    ```
5. Enter the virtual environment:
   ```bash
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```
### Manually
#### Using UV (recommended)
1. Clone this repository
2. Navigate to the project directory 
3. Create a virtual environment:
   ```bash
   uv venv
   ```
4. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```
5. Install the required dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```
#### Using python virtual environment
1. Clone this repository
2. Navigate to the project directory
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
4. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
#### Without virtual environment
1. Clone this repository
2. Navigate to the project directory
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
```bash
python cli_app.py
```
Input the email text and one a new line write `EOF` or `DONE` and press `enter` to proceed to the prediction.
![Phishing Classifier Input Page](./images/Screenshot 2025-05-01 at 13.40.16.png)
![Phishing Classifier Input Page, with an example email](./images/Screenshot 2025-05-01 at 13.41.49.png)
The result will be displayed in the terminal.
![Phishing Classifier Result Page](./images/Screenshot 2025-05-01 at 13.42.28.png)
You can press `enter` to analyze another email or exit the app by typing `exit` or `quit` and then pressing `enter`. 

## Training the Model (Optional)
If you want to retrain the model:
1. Navigate to the training_script directory
2. Run:
   ```bash
   python train_model.py
   ```

## Project Structure
- `cli_app.py`: Main CLI application
- `training_script/train_model.py`: Model training script
- `trained_model/`: Contains pre-trained model files
- `requirements.txt`: Python dependencies

## License
I don't any idea about the licensing, if you want to fork it, feel free to do so and use it for your own purposes. I take no responsibility for any misuse of this code.