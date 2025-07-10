import os
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file
import torch

# --- Paths ---
# Relative path to model directory - adjust based on your setup
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "proxy_models_for_intel", "test_bigger_model")
# Alternative: Use test_smaller_model for smaller model
# BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "proxy_models_for_intel", "test_smaller_model")
SAFE_PATH = os.path.join(BASE_DIR, "model.safetensors")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
ONNX_PATH = os.path.join(BASE_DIR, "model.onnx")

# --- Load config ---
config = AutoConfig.from_pretrained(CONFIG_PATH)

# --- Dynamically choose the right model class ---
from transformers import (
    AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification,
    AutoModelForQuestionAnswering, AutoModelForMaskedLM
)

# Get the architecture from config (e.g., "BertForSequenceClassification")
architecture = config.architectures[0] if hasattr(config, "architectures") else None

# Map architecture to class
model_class = {
    "BertModel": AutoModel,
    "BertForSequenceClassification": AutoModelForSequenceClassification,
    "BertForTokenClassification": AutoModelForTokenClassification,
    "BertForQuestionAnswering": AutoModelForQuestionAnswering,
    "BertForMaskedLM": AutoModelForMaskedLM,
    # Add more as needed for other architectures
}.get(architecture, AutoModel)  # Default to AutoModel

# --- Load model weights from safetensors ---
# --- Load model weights from safetensors ---
state_dict = load_file(SAFE_PATH)
model = model_class.from_config(config)          # Instantiate from config
model.load_state_dict(state_dict)       # Load weights
model.eval()

# --- Prepare Dummy Input ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Or use your model name if not BERT
dummy_text = "ONNX export test input."
inputs = tokenizer(dummy_text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
input_names = list(inputs.keys())
dummy_input = tuple(inputs.values())

# --- Export to ONNX ---
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=input_names,
    output_names=["logits"],  # For BERT classifiers, usually 'logits'
    opset_version=14,
    do_constant_folding=True,
    export_params=True,
    training=torch.onnx.TrainingMode.EVAL,
    verbose=False
)

print(f"Exported ONNX model to {ONNX_PATH}")
