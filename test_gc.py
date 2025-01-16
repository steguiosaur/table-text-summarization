import os
from rouge_score import rouge_scorer, scoring

def concatenate_predictions(tabletotext_path, summarizedt_path, gcsummarize_path, split):
    # Define the input and output directories
    tabletotext_pred_path = os.path.join(tabletotext_path, f'predictions_{split}/')
    summarizedt_pred_path = os.path.join(summarizedt_path, f'predictions_{split}/')
    gcsummarize_pred_path = os.path.join(gcsummarize_path, f'predictions_{split}/')

    # Create the output directory if it doesn't exist
    os.makedirs(gcsummarize_pred_path, exist_ok=True)

    # Concatenate predictions from `tabletotext` and `summarizedt` for each file
    for i in range(50):  # Assuming files are named from 0 to 49
        tabletotext_file = os.path.join(tabletotext_pred_path, f"{i}_prediction.txt")
        summarizedt_file = os.path.join(summarizedt_pred_path, f"{i}_prediction.txt")
        gcsummarize_file = os.path.join(gcsummarize_pred_path, f"{i}_prediction.txt")

        # Read the contents of both files and concatenate them
        with open(tabletotext_file, 'r') as f1, open(summarizedt_file, 'r') as f2:
            text1 = f1.read().strip()
            text2 = f2.read().strip()
            concatenated_text = text1 + " " + text2

        # Write the concatenated text to `gcsummarize` predictions directory
        with open(gcsummarize_file, 'w') as f3:
            f3.write(concatenated_text)

def validation_task(split="test"):
    log_path = "/content/drive/MyDrive/Output/logs/d2t/outputs"
    tabletotext = os.path.join(log_path, "tbl2textnolf_test")
    summarizedt = os.path.join(log_path, "only_text")
    gcsummarize = os.path.join(log_path, "gc_test")

    # Ensure output directories exist
    if not os.path.exists(gcsummarize):
        os.makedirs(gcsummarize)

    # Concatenate predictions from `tabletotext` and `summarizedt` into `gcsummarize`
    concatenate_predictions(tabletotext, summarizedt, gcsummarize, split)

    # Paths for the reference and concatenated predictions for ROUGE evaluation
    ref_split = os.path.join(summarizedt, f'references_{split}/')
    pred_split = os.path.join(gcsummarize, f'predictions_{split}/')
    
    # Generate single-file references and predictions for ROUGE scoring
    gt_path = os.path.join(gcsummarize, f'references_{split}.txt')
    pred_path = os.path.join(gcsummarize, f'predictions_{split}.txt')

    # Aggregate reference files into a single reference text file
    with open(gt_path, 'w') as gt_file:
        for i in range(50):  # Assuming reference files range from 0 to 49
            ref_file = os.path.join(ref_split, f"{i}_reference.txt")
            with open(ref_file, 'r') as rf:
                gt_file.write(rf.read().strip() + "\n")

    # Aggregate concatenated prediction files into a single prediction text file
    with open(pred_path, 'w') as pred_file:
        for i in range(50):  # Assuming prediction files range from 0 to 49
            pred_file_path = os.path.join(pred_split, f"{i}_prediction.txt")
            with open(pred_file_path, 'r') as pf:
                pred_file.write(pf.read().strip() + "\n")

    # ROUGE evaluation using `pyrouge`
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    with open(gt_path, 'r') as gt_file, open(pred_path, 'r') as pred_file:
        references = gt_file.readlines()
        predictions = pred_file.readlines()

        for ref, pred in zip(references, predictions):
            ref = ref.strip()
            pred = pred.strip()

            # Compute ROUGE scores for each sample
            scores = scorer.score(ref, pred)
            aggregator.add_scores(scores)

            # Log per-sample ROUGE scores
            print("Iteration ROUGE Scores:")
            for rouge_type, score in scores.items():
                print(f"{rouge_type}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1={score.fmeasure:.4f}")

    # Compute and log overall ROUGE scores
    result = aggregator.aggregate()
    overall_scores = {}
    for rouge_type, value in result.items():
        overall_scores[rouge_type] = {
            "Precision": value.mid.precision * 100,
            "Recall": value.mid.recall * 100,
            "F1": value.mid.fmeasure * 100
        }

    print("\n[INFO] Overall ROUGE Scores:")
    for rouge_type, scores in overall_scores.items():
        print(f"{rouge_type}: Precision: {scores['Precision']:.2f}, Recall: {scores['Recall']:.2f}, F1: {scores['F1']:.2f}")

    return overall_scores
# Example usage
validation_task("test")
