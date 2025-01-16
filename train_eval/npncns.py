import json
import os
from tqdm import tqdm
import argparse
import re

def extract_numbers(text):
    """Extract numbers from text"""
    return [float(num) for num in re.findall(r'\d+(?:\.\d+)?', text)]

def M(set1, set2):
    """Count common elements in two sets"""
    return len(set(set1) & set(set2))

def calculate_np(H_n, S_n):
    """Calculate Number Precision"""
    if len(H_n) == 0:
        return 0
    return M(H_n, S_n) / len(H_n)

def calculate_nc(D_n, H_n, S_n):
    """Calculate Number Coverage"""
    if len(S_n) == 0:
        return 0
    
    nr = M(H_n, S_n) / len(S_n)
    
    # Check if the intersection is empty
    if not set(D_n).isdisjoint(set(S_n)):
        return nr * len(S_n) / M(D_n, S_n)
    else:
        return 0

def calculate_ns(np, nc):
    """Calculate Number Selection"""
    if np + nc == 0:
        return 0
    return 2 * np * nc / (np + nc)

# def calculate_np(H_n, S_n):
#     """Calculate Number Precision"""
#     return M(H_n, S_n) / len(H_n)
#
# def calculate_nc(D_n, H_n, S_n):
#     """Calculate Number Coverage"""
#     nr = M(H_n, S_n) / len(S_n)
#     return nr * len(S_n) / M(D_n, S_n)
#
# def calculate_ns(np, nc):
#     """Calculate Number Selection"""
#     return 2 * np * nc / (np + nc)

def evaluate_summary(D, S, H):
    """
    Evaluate a summary based on numerical information usage
    D: Input document
    S: Target summary
    H: Generated summary
    """
    D_n = extract_numbers(D)
    S_n = extract_numbers(S)
    H_n = extract_numbers(H)
    
    np = calculate_np(H_n, S_n)
    nc = calculate_nc(D_n, H_n, S_n)
    ns = calculate_ns(np, nc)
    
    return np, nc, ns

def prepare_input(data):
    """Prepare input by concatenating relevant fields"""
    caption = data.get("table_caption", "")
    columns = " ".join(data.get("table_column_names", []))
    content = " ".join([item for sublist in data.get("table_content_values", []) for item in sublist])
    long_text = data.get("long_text", "")
    
    return " ".join([caption, columns, content, long_text])

def main():
    parser = argparse.ArgumentParser(description='Evaluate summarization task')
    parser.add_argument('--affix_model_name', type=str, required=True, help='Model name affix')
    args = parser.parse_args()

    # Paths
    pred_path = f"/content/drive/MyDrive/Output/logs/d2t/outputs/{args.affix_model_name}/predictions_test.txt"
    ref_path = f"/content/drive/MyDrive/Output/logs/d2t/outputs/{args.affix_model_name}/references_test.txt"
    input_data_path = "/content/drive/MyDrive/Dataset/SciGenMod/test-50.json"

    # Load input data
    with open(input_data_path, 'r') as f:
        input_data = json.load(f)

    # Load predictions and references
    with open(pred_path, 'r') as f_pred, open(ref_path, 'r') as f_ref:
        predictions = f_pred.readlines()
        references = f_ref.readlines()

    # Ensure we have the same number of samples
    assert len(predictions) == len(references) == len(input_data), "Number of samples mismatch"

    np_scores = []
    nc_scores = []
    ns_scores = []

    for i in tqdm(range(len(predictions))):
        pred_text = predictions[i].strip()
        ref_text = references[i].strip()
        
        # Prepare input text
        input_id = str(i + 400)  # Assuming IDs start from 400
        input_text = prepare_input(input_data[input_id])
        
        np, nc, ns = evaluate_summary(input_text, ref_text, pred_text)
        np_scores.append(np)
        nc_scores.append(nc)
        ns_scores.append(ns)

    avg_np = sum(np_scores) / len(np_scores)
    avg_nc = sum(nc_scores) / len(nc_scores)
    avg_ns = sum(ns_scores) / len(ns_scores)

    print(f"Average NP score: {avg_np:.4f}")
    print(f"Average NC score: {avg_nc:.4f}")
    print(f"Average NS score: {avg_ns:.4f}")

if __name__ == "__main__":
    main()
