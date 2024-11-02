# Preprocessing
# - Filter syntax of HTML, Markdown, and LaTeX using regex
# - Tokenization
# - Case-folding (lowercase)


import re
import json
from collections import defaultdict
import pandas as pd


class Preprocess:
    @staticmethod
    def parse_markdown(text: str) -> dict:
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

        return extracted_data
        # return json.dumps(extracted_data, ensure_ascii=False, indent=4)

    @staticmethod
    def linearize_table_data(data: dict) -> dict:
        # Extract the json structure
        table_caption = data["table_caption"]
        table_header = data["table_column_names"]
        table_contents = data["table_content_values"]
        textual_data = data["long_text"]

        # Initialize source text with the caption
        src_text = f"<table> <caption> {table_caption} </caption> "

        # If logical type information is needed
        #if add_type:
        #    for seg in data['segmented_text']:
        #        src_text += f"<type> {seg['action']} </type> {seg['text']} "

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

        # Add dynamically computed aggregation values if needed
        #if pre_com:
        #    sum_vals = pd_table.sum(numeric_only=True)
        #    avg_vals = pd_table.mean(numeric_only=True)
        #
        #    if not sum_vals.empty:
        #        for col, val in sum_vals.items():
        #            src_text += f"<sum_cell> {val} <col_header> {col} </col_header> </sum_cell> "
        #
        #    if not avg_vals.empty:
        #        for col, val in avg_vals.items():
        #            src_text += f"<avg_cell> {val} <col_header> {col} </col_header> </avg_cell> "

        # Add caption at the end of the table linearization
        src_text += f"</table> {textual_data}"

        # Assign linearized text to 'src_text' field in data
        data['src_text'] = src_text

        return data

    # @staticmethod
    # def tokenize_latex_table(text: str) -> list:
    #     # remove leading and trailing whitespace
    #     text = text.strip()
    #
    #     # check if the text contains a LaTeX table
    #     if "\\begin{tabular}" not in text or "\\end{tabular}" not in text:
    #         return []  # invalid LaTeX table format
    #
    #     # extract table content
    #     table_content = re.search(r"\\begin{tabular}.+?\\end{tabular}", text, re.DOTALL)
    #     if table_content is None:
    #         return []  # unable to find valid table content
    #     table_text = table_content.group(0)
    #
    #     # split cells based on columns and rows
    #     cells = re.split(r"(?<=&)(?=[^&]*\\\\|$)", table_text)
    #
    #     # split cells based on columns and rows
    #     words = []
    #     for cell in cells:
    #         # split cells into words based on spaces
    #         cell_words = cell.strip().split()
    #         # filter LaTeX reserved words in text and convert to lowercase
    #         cell_words = [
    #             word.lower()
    #             for word in cell_words
    #             if not (word.startswith("\\") or word == "&")
    #         ]
    #         words.extend(cell_words)
    #
    #     return words
    #
    # @staticmethod
    # def tokenize_html_table(html: str) -> list:
    #     # parse HTML using BeautifulSoup
    #     soup = BeautifulSoup(html, "html.parser")
    #
    #     # find all table rows and cells
    #     table = soup.find("table")
    #     if not table:
    #         return []  # no table found
    #
    #     words = []
    #     for row in table.find_all("tr"):
    #         cells = row.find_all(["td", "th"])
    #         for cell in cells:
    #             # get text content
    #             cell_text = cell.get_text(separator=" ", strip=True)
    #             # convert to lowercase and split to words
    #             cell_words = cell_text.lower().split()
    #             words.extend(cell_words)
    #
    #     return words
    #
    #
#
# # Example usage:
# markdown_table = """
# | Fruit    | Quantity |
# |----------|----------|
# | Apple    | 10       |
# | Orange   | 5        |
# | Banana   | 8        |
# """
#
# latex_table = """
# \\begin{tabular}{|c|c|}
# \\hline
# Fruit & Quantity \\\\
# \\hline
# Apple & 10 \\\\
# Orange & 5 \\\\
# Banana & 8 \\\\
# \\hline
# \\end{tabular}
# """
#
# html_table = """
# <table>
#   <tr>
#     <th>Fruit</th>
#     <th>Quantity</th>
#   </tr>
#   <tr>
#     <td>Apple</td>
#     <td>10</td>
#   </tr>
#   <tr>
#     <td>Orange</td>
#     <td>5</td>
#   </tr>
#   <tr>
#     <td>Banana</td>
#     <td>8</td>
#   </tr>
# </table>
# """
#
# print(Tokenizer().tokenize_markdown_table(markdown_table))
# print(Tokenizer().tokenize_latex_table(latex_table))
# print(Tokenizer().tokenize_html_table(html_table))
