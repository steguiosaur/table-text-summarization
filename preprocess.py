import json
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import re

def extract_markdown_data(text):
    # Match the table caption
    caption_match = re.search(r'^(Table \d+: .+?)\n', text, re.MULTILINE)
    table_caption = caption_match.group(1) if caption_match else None
    
    # Extract column headers
    column_names_match = re.search(r'^\|\s*(.+?)\s*\|\n\|\s*([-:| ]+)\n', text, re.MULTILINE)
    table_column_names = [col.strip() for col in column_names_match.group(1).split('|')] if column_names_match else []
    
    # Extract table rows
    rows_matches = re.findall(r'^\|\s*(.+?)\s*\|$', text, re.MULTILINE)
    # Modify the table row extraction to exclude separator lines
    table_content_values = [
        [cell.strip() for cell in row.split('|')]
        for row in rows_matches[1:]  # skip header row
        if not all(cell.strip() == '-' * len(cell.strip()) for cell in row.split('|'))  # exclude separator
    ]

    # Extract surrounding long text by removing table content
    long_text = re.sub(r'\n\|.*?\|$', '', text, flags=re.MULTILINE).strip()
    if caption_match:
        long_text = long_text.replace(caption_match.group(0), "").strip()
    long_text = re.sub(r'\n', ' ', long_text).strip()

    # Structure the output in JSON format
    extracted_data = {
        "table_caption": table_caption,
        "table_column_names": table_column_names,
        "table_content_values": table_content_values,
        "long_text": long_text
    }
    
    return json.dumps(extracted_data, ensure_ascii=False, indent=4)


def linearize_table_data(data: dict, add_type: bool = False, pre_com: bool = False) -> dict:
    # Extract the json structure
    table_caption = data["table_caption"]
    table_header = data["table_column_names"]
    table_contents = data["table_content_values"]
    textual_data = data["long_text"]

    # Initialize source text with the caption
    src_text = f"<table> <caption> {str(table_caption)} </caption> "

    # convert table data to pandas for easier indexing
    pd_in = defaultdict(list)
    for ind, header in enumerate(table_header):
        for row in table_contents:
            pd_in[header].append(row[ind])
    pd_table = pd.DataFrame(pd_in)

    # Linearize the entire table (exclude highlights for now)
    for row_idx in range(len(pd_table)):
        for col_header in pd_table.columns:
            cell_value = pd_table[col_header].iloc[row_idx]
            # Construct cell string for each cell in the table
            cell_str = f"<cell> {cell_value} <col_header> {col_header} </col_header> <row_idx> {row_idx} </row_idx> </cell> "
            src_text += cell_str

    # Add caption at the end of the table linearization
    src_text += f"</table> {textual_data}"

    # Assign linearized text to 'src_text' field in data
    data['src_text'] = src_text

    return data


def preprocess(text):
    json_md = extract_markdown_data(text)
    json_dict = json.loads(json_md)
    return linearize_table_data(json_dict)

input_data = """
Emotion Recognition Performance Analysis

Our recent experiment with emotion recognition in natural language processing yielded intriguing results.
We trained a deep learning model to classify emotional states expressed in text across six categories: anger, disgust, fear, joy, sadness, and surprise.
The model's performance was evaluated using precision, recall, and F1-score metrics for each emotion category.

Notably, the model struggled with recognizing anger and surprise, achieving lower F1-scores compared to other emotions.
Anger detection resulted in an F1-score of 0.621, while surprise recognition scored slightly higher at 0.663.
These findings suggest that the model may benefit from additional training data or fine-tuning to improve its ability to distinguish between these nuanced emotional states.

On the other hand, the model performed exceptionally well in identifying joy, with an impressive F1-score of 0.783.
This high accuracy could be attributed to the distinctive linguistic patterns associated with joyful expressions in text.
Fear detection also showed strong results, with an F1-score of 0.732, indicating the model's effectiveness in capturing subtle cues of apprehension or anxiety in language.
Interestingly, the model demonstrated balanced performance across precision and recall for most emotions, except for surprise, where recall outperformed precision.
This discrepancy warrants further investigation into the model's tendency to over-predict surprise in certain contexts.

Overall, our model achieved an average F1-score of 0.694 across all emotion categories, suggesting a solid foundation for emotion recognition tasks.
However, the varying levels of success across different emotions highlight areas for improvement and opportunities for targeted optimization in future iterations of the model.

| Emotion  | Precision | Recall | F1-score |
| -------- | --------- | ------ | -------- |
| anger    | 0.643     | 0.601  | 0.621    |
| disgust  | 0.703     | 0.661  | 0.682    |
| fear     | 0.742     | 0.721  | 0.732    |
| joy      | 0.762     | 0.805  | 0.783    |
| sad      | 0.685     | 0.661  | 0.673    |
| surprise | 0.627     | 0.705  | 0.663    |
| Average  | 0.695     | 0.695  | 0.694    |

"""

json_data = preprocess(input_data)
print(json.dumps(json_data, indent=4))
