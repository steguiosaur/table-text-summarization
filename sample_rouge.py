from rouge_score import rouge_scorer

# Sample summaries
target_summary = (
    "The study highlights the effectiveness of retrieval-based chatbot models, "
    "emphasizing the impact of dataset size and the number of negative samples "
    "on performance. AUC scores improved significantly with more negative samples."
)

generated_summary = (
    "Retrieval-based chatbot models are effective, with larger datasets and "
    "more negative samples enhancing performance. The study showed improvements "
    "in AUC scores with increased negative samples."
)

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# Calculate scores
scores = scorer.score(target_summary, generated_summary)

# Display the scores
print("ROUGE Scores:")
for rouge_type, score in scores.items():
    print(f"\n{rouge_type.upper()}:")
    print(f"  Precision: {score.precision:.4f}")
    print(f"  Recall:    {score.recall:.4f}")
    print(f"  F1 Score:  {score.fmeasure:.4f}")

