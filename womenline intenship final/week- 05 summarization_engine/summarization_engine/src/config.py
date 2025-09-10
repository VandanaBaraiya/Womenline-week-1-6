import os

MODEL_NAME = os.environ.get("MODEL_NAME", "facebook/bart-large-cnn")
MAX_SUMMARY_SENTENCES = int(os.environ.get("MAX_SUMMARY_SENTENCES", "3"))
DEVICE = os.environ.get("DEVICE")  # set 'cpu' or 'cuda'; if None, auto
