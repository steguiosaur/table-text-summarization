import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ModelLoader:
    def __init__(self, model, model_path=None):
        self.model_name = model
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def tntsumm(self):
        # Load tokenizer and add custom tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        markers = ["{", "}", "<table>", "</table>", "<type>", "</type>", "<cell>", "</cell>", "<col_header>", "</col_header>", "<row_idx>", "</row_idx>"]
        markers += ["<max_rank>", "</max_rank>", "<min_rank>", "</min_rank>", "<sum_cell>", "</sum_cell>", "<avg_cell>", "</avg_cell>"]
        self.tokenizer.add_tokens(markers)

        # Load model and resize embeddings to include custom tokens
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Load fine-tuned weights if provided
        if self.model_path:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        # Move model to specified device
        self.model.to(self.device)
        self.model.eval()  # Set to inference mode

        return self.model, self.tokenizer

    def generate_output(self, input_text, max_length=200):
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded first.")

        # Tokenize the input text and move to the specified device
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate output tokens
        with torch.no_grad():  # No need to compute gradients during inference
            output_ids = self.model.generate(**inputs, max_length=max_length)

        # Decode the generated output tokens into text
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return output_text

