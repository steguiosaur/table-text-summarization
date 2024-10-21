# Preprocessing
# - Filter syntax of HTML, Markdown, and LaTeX using regex
# - Tokenization
# - Case-folding (lowercase)


import re
from bs4 import BeautifulSoup


class Tokenizer:
    @staticmethod
    def tokenize_markdown_table(text: str) -> list:
        # remove leading and trailing whitespace
        text = text.strip()

        # validate if text contains markdown table
        if "|" not in text or "-" not in text:
            return []  # not valid Markdown table format

        # extract only the table content and not the separators
        lines = text.splitlines()
        table_lines = [
            line
            for line in lines
            if "|" in line
            and not (
                line.strip().startswith("| -")
                or line.strip().startswith("|-")
                or line.strip().startswith("| :")
                or line.strip().startswith("|:")
            )
        ]

        # collect the words in the table
        words = []
        for line in table_lines:
            cells = re.split(r"\s*\|\s*", line)
            for cell in cells:
                if cell.strip():  # ignore empty cells
                    # split cells into words based on spaces
                    cell_words = cell.strip().split()
                    # convert each word to lowercase
                    cell_words = [word.lower() for word in cell_words]
                    words.extend(cell_words)

        return words

    @staticmethod
    def tokenize_latex_table(text: str) -> list:
        # remove leading and trailing whitespace
        text = text.strip()

        # check if the text contains a LaTeX table
        if "\\begin{tabular}" not in text or "\\end{tabular}" not in text:
            return []  # invalid LaTeX table format

        # extract table content
        table_content = re.search(r"\\begin{tabular}.+?\\end{tabular}", text, re.DOTALL)
        if table_content is None:
            return []  # unable to find valid table content
        table_text = table_content.group(0)

        # split cells based on columns and rows
        cells = re.split(r"(?<=&)(?=[^&]*\\\\|$)", table_text)

        # split cells based on columns and rows
        words = []
        for cell in cells:
            # split cells into words based on spaces
            cell_words = cell.strip().split()
            # filter LaTeX reserved words in text and convert to lowercase
            cell_words = [
                word.lower()
                for word in cell_words
                if not (word.startswith("\\") or word == "&")
            ]
            words.extend(cell_words)

        return words

    @staticmethod
    def tokenize_html_table(html: str) -> list:
        # parse HTML using BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # find all table rows and cells
        table = soup.find("table")
        if not table:
            return []  # no table found

        words = []
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            for cell in cells:
                # get text content
                cell_text = cell.get_text(separator=" ", strip=True)
                # convert to lowercase and split to words
                cell_words = cell_text.lower().split()
                words.extend(cell_words)

        return words


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
