import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model_and_tokenizer(model_name, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    markers = ["{", "}", "<table>", "</table>", "<type>", "</type>", "<cell>", "</cell>", "<col_header>", "</col_header>", "<row_idx>", "</row_idx>"]
    markers += ["<max_rank>", "</max_rank>", "<min_rank>", "</min_rank>", "<sum_cell>", "</sum_cell>", "<avg_cell>",
                "</avg_cell>"]
    tokenizer.add_tokens(markers)

    # load the model (with the pretrained or fine-tuned weights)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.resize_token_embeddings(len(tokenizer))

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))  # Load fine-tuned weights

    # Set to evaluation mode
    model.eval()
    return model, tokenizer

def generate_output(model, tokenizer, input_text, device='cpu', max_length=200):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # Move inputs to the specified device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Generate output tokens
    with torch.no_grad():  # No need to compute gradients during inference
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)

    # Decode the generated output tokens into text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

if __name__ == '__main__':
    # Example usage

    # Load the model and tokenizer
    model_name = 'facebook/bart-large'  # Change to your specific model name
    # model_path = "/content/drive/MyDrive/Output/models/d2t/text_train/bart-large_ep1.pt"
    model_path = "~/Downloads/bart-large_ep1(1).pt"

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, model_path)

    # Example input (this could be a table structure or any other input your model was trained on)
    input_text = "<table> <cell> model 1 </cell> <cell> 54 </cell> </table> <table> <cell> model 2 </cell> <cell> 23 </cell> </table>  Rouge score of model 1 is greater than model 2. This means we could interpret that model 1 could produce a more targetted summary."


    # Generate output
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    output_text = generate_output(model, tokenizer, input_text, device)

    print(f"Generated Output: {output_text}")
