import logging
import os
import sys
import re
import time  
import argparse
import pyfiglet

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertModel
from safetensors.torch import load_file as safe_load_file

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align

console = Console()

PROCESSING_ASCII = pyfiglet.figlet_format("PROCESSING", font="slant")
LEGIT_ASCII = pyfiglet.figlet_format("LEGIT", font="slant")
PHISHING_ASCII = pyfiglet.figlet_format("PHISHING", font="slant")


def clean_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s\.\?,!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class DistilBertLSTMClassifier(nn.Module):
    def __init__(self, distilbert_model_name='distilbert-base-uncased', lstm_hidden_size=128, lstm_layers=1, dropout_rate=0.2, num_labels=1):
        super().__init__()
        self.num_labels = num_labels
        self.distilbert = DistilBertModel.from_pretrained(distilbert_model_name)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(input_size=self.distilbert.config.hidden_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout_rate if lstm_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = distilbert_output.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(lstm_output.size()).float()
        sum_embeddings = torch.sum(lstm_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        dropped_output = self.dropout(pooled_output)
        logits = self.classifier(dropped_output)
        return {"logits": logits}

def predict_phishing(text, model, tokenizer, device, max_length=128):
    cleaned_text = clean_text(text)
    inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs['logits']
    probability = torch.sigmoid(logits).squeeze().item()
    prediction = 1 if probability > 0.5 else 0
    return prediction, probability

def load_model_and_tokenizer(model_path, base_model_name='distilbert-base-uncased'):
    if not os.path.isdir(model_path):
        console.print(Align.center(f"[bold red]Error:[/bold red] Model directory not found at the required path: {model_path}"))
        console.print(Align.center("[bold red]Please ensure the model files are located at '../phishing_results/final_model' relative to the script.[/bold red]"))
        sys.exit(1)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    except Exception as e_pretrained:
        console.print(Align.center(f"[yellow]Warning:[/yellow] Failed to load tokenizer with from_pretrained ({e_pretrained}). Trying explicit file loading..."))
        try:
            vocab_file = os.path.join(model_path, "vocab.txt")
            tokenizer_file = os.path.join(model_path, "tokenizer.json")

            if not os.path.exists(vocab_file):
                 console.print(Align.center(f"[bold red]Error:[/bold red] vocab.txt not found in {model_path}"))
                 sys.exit(1)

            tok_kwargs = {"vocab_file": vocab_file}
            if os.path.exists(tokenizer_file):
                tok_kwargs["tokenizer_file"] = tokenizer_file

            tokenizer = DistilBertTokenizerFast(**tok_kwargs)
        except Exception as e_explicit:
            console.print(Align.center(f"[bold red]Error:[/bold red] Failed to load tokenizer explicitly: {e_explicit}"))
            sys.exit(1)

    try:
        model_structure = DistilBertLSTMClassifier(distilbert_model_name=base_model_name, num_labels=1)

        state_dict_path = os.path.join(model_path, "model.safetensors")
        state_dict_path_pytorch = os.path.join(model_path, "pytorch_model.bin")

        if not os.path.exists(state_dict_path):
             if not os.path.exists(state_dict_path_pytorch):
                 console.print(Align.center(f"[bold red]Error:[/bold red] Neither model.safetensors nor pytorch_model.bin found in {model_path}"))
                 sys.exit(1)
             else:
                 state_dict_path = state_dict_path_pytorch
                 console.print(Align.center(f"[yellow]Warning:[/yellow] Using pytorch_model.bin instead of model.safetensors."))
                 state_dict = torch.load(state_dict_path, map_location="cpu")
        else:
             state_dict = safe_load_file(state_dict_path, device="cpu")

        has_model_prefix = all(k.startswith('model.') for k in state_dict.keys())
        needs_prefix_removal = has_model_prefix and not any(k.startswith('model.') for k in model_structure.state_dict().keys())

        if needs_prefix_removal:
             console.print(Align.center("[yellow]Adjusting state_dict keys (removing 'model.' prefix)...[/yellow]"))
             state_dict = {k.partition('model.')[2]: v for k, v in state_dict.items()}

        missing_keys, unexpected_keys = model_structure.load_state_dict(state_dict, strict=False)

        if unexpected_keys:
            console.print(Align.center(f"[yellow]Warning:[/yellow] Unexpected keys in state_dict: {unexpected_keys}"))
        if missing_keys:
            is_just_classifier = all(k.startswith('classifier.') for k in missing_keys)
            if not is_just_classifier:
                 console.print(Align.center(f"[bold red]Error:[/bold red] Missing non-classifier keys in state_dict: {missing_keys}. Model architecture might mismatch."))
            else:
                 console.print(Align.center(f"[yellow]Warning:[/yellow] Missing classifier keys - likely okay if base model loaded: {missing_keys}"))

        model_structure.to(device)
        return model_structure, tokenizer, device

    except Exception as e:
        console.print(Align.center(f"[bold red]Error loading model:[/bold red] {e}"))
        import traceback
        traceback.print_exc()
        sys.exit(1)

def clear_screen():
    console.clear()

def input_page():
    clear_screen()
    title = Align.center("[bold cyan]Phishing Detector CLI[/bold cyan]")
    panel = Panel(
        Align.center("[italic]Enter or paste the email text below. Type 'EOF' or 'DONE' on a new line when finished.[/italic]"),
        title="Input Email Text",
        border_style="cyan",
        width=80,
        padding=(1, 2)
    )
    console.print("\n")
    console.print(title)
    console.print("\n")
    console.print(Align.center(panel))
    console.print(Align.left("> "), end="")

    user_input_lines = []
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line_stripped = line.strip().upper()
            if line_stripped == 'EOF' or line_stripped == 'DONE':
                break
            user_input_lines.append(line)
        except KeyboardInterrupt:
            console.print("\n[yellow]Input cancelled.[/yellow]")
            return None # Indicate cancellation

    user_input = "".join(user_input_lines).strip()
    if not user_input:
         console.print(Align.center("[yellow]No input received.[/yellow]"))
         return None
    return user_input

def processing_page():
    clear_screen()
    panel = Panel(
        Text(PROCESSING_ASCII, style="bold yellow", justify="center"),
        border_style="yellow",
        width=80,
        padding=(1, 2)
    )
    console.print("\n")
    console.print(Align.center(panel))
    console.print("\n")

def response_page(prediction, probability):
    clear_screen()

    if prediction == 1:
        result_text = "PHISHING"
        ascii_art = PHISHING_ASCII
        style = "bold red"
        border_style = "red"
    else:
        result_text = "SAFE EMAIL"
        ascii_art = LEGIT_ASCII
        style = "bold green"
        border_style = "green"

    panel = Panel(
        Text(ascii_art, style=style, justify="center"),
        border_style=border_style,
        width=80,
        padding=(1, 2)
    )

    console.print("\n")
    console.print(Align.center(panel))
    console.print("\n")
    console.print(Align.center(f"Prediction: [b]{result_text}[/b]"))
    console.print(Align.center(f"Confidence Score (Probability of Phishing): [b]{probability:.4f}[/b]"))
    console.print("\n")
    console.print(Align.center("[italic]Press Enter to analyze another email, or type 'quit' or 'exit' to end.[/italic]"))
    try:
        user_action = input("> ").strip().lower()
        if user_action in ['quit', 'exit']:
            console.print("\n[bold cyan]Exiting application.[/bold cyan]")
            return False
        return True
    except KeyboardInterrupt:
        console.print("\n[bold cyan]Exiting application (Ctrl+C detected).[/bold cyan]")
        return False

def main(fixed_model_path, base_model_name):
    MIN_PROCESSING_DISPLAY_TIME = 1.0

    try:
        model, tokenizer, device = load_model_and_tokenizer(fixed_model_path, base_model_name)
    except SystemExit:
        return

    try:
        while True:
            email_text = input_page()
            if email_text is None:
                clear_screen()
                console.print(Align.center("[yellow]Input cancelled or empty.[/yellow]"))
                action = input("Press Enter to try again, or type 'quit'/'exit' to end: ").strip().lower()
                if action in ['quit', 'exit']:
                    console.print("\n[bold cyan]Exiting application.[/bold cyan]")
                    break
                else:
                    continue

            processing_page()

            start_time = time.time()

            prediction, probability = predict_phishing(email_text, model, tokenizer, device)

            end_time = time.time()
            elapsed_time = end_time - start_time

            if elapsed_time < MIN_PROCESSING_DISPLAY_TIME:
                sleep_duration = MIN_PROCESSING_DISPLAY_TIME - elapsed_time
                time.sleep(sleep_duration)

            continue_running = response_page(prediction, probability)
            if not continue_running:
                break

    except KeyboardInterrupt:
        console.print("\n" + Align.center("[bold yellow]Program interrupted. Exiting...[/bold yellow]"))
    except Exception as e:
        console.print(Align.center(f"\n[bold red]An unexpected error occurred:[/bold red] {e}"))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir:
        fixed_model_rel_path = "./trained_model/final_model"
        fixed_model_abs_path = os.path.abspath(os.path.join(script_dir, fixed_model_rel_path))
    else:
        console.print("[bold red]Error:[/bold red] Could not determine script directory. Path to the model might be incorrect.")
        fixed_model_abs_path = os.path.abspath("./phishing_results/final_model")
        console.print(f"[yellow]Warning:[/yellow] Assuming model path relative to current directory: {fixed_model_abs_path}")


    parser = argparse.ArgumentParser(
        description="CLI Tool to detect phishing emails using a pre-trained model (fixed path)."
    )
    parser.add_argument(
        "-b", "--base_model",
        type=str,
        default="distilbert-base-uncased",
        help="Base model name for the DistilBERT model. Default is 'distilbert-base-uncased'."        
    )

    parsed_args = parser.parse_args()

    main(fixed_model_abs_path, parsed_args.base_model)