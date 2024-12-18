# T&TSumm

Codes, documentations, and datasets for the paper "Faithfulness in Textual and
Tabular Data Summarizers Using Logical Form Representation"

## Expected Model Explanation

A table is defined as $T = \{ T_{ij} : 1 \leq i \leq R_T, 1 \leq j \leq R_C \}$
where $R_T$ is the number of rows and $R_C$ is the number of columns.
$T_{ij}$ is the value in row $i$ and column $j$ of the table.

Model should output a **summary** $y$ given the input **textual data** (X),
**tabular data** $T$ and **highlighted cells** $H$. Highlights $H$ are evaluated
through the content selection step where a table data value $T_{ij}$ is present
in textual data $X$, returning the row $i$ and column $j$ values of cells.

$$h = \{(i, j)\}$$
$$H = \{h_1, h_2, \ldots, h_n\}$$
$$P(y | X, T ; H)$$

## Setup

Currently working on `python 3.12.7`

Install dependencies using `pip install -r requirements.txt`

```python
torch==2.4.1
transformers==4.45.2
pandas==2.2.3
nltk==3.9.1
```

For evaluation, do not install `pyrouge` using pip. Follow this
[guide](https://stackoverflow.com/a/57686103/20493501) or use the
following command if you are on google colab.

```sh
!git clone https://github.com/bheinzerling/pyrouge /content/pyrouge/
!pip install -e /content/pyrouge/
!git clone https://github.com/andersjo/pyrouge.git /content/pyrouge/rouge
!pyrouge_set_rouge_path /content/pyrouge/rouge/tools/ROUGE-1.5.5/
!apt-get install -y libxml-parser-perl
!rm /content/pyrouge/rouge/tools/ROUGE-1.5.5/data/WordNet-2.0.exc.db
!/content/pyrouge/rouge/tools/ROUGE-1.5.5/data/WordNet-2.0-Exceptions/buildExeptionDB.pl /content/pyrouge/rouge/tools/ROUGE-1.5.5/data/WordNet-2.0-Exceptions /content/pyrouge/rouge/tools/ROUGE-1.5.5/data/smart_common_words.txt /content/pyrouge/rouge/tools/ROUGE-1.5.5/data/WordNet-2.0.exc.db
!python -m pyrouge.test
```

## Training

Access executor and dataset from:

- [Colab](https://colab.research.google.com/drive/1bjb6SYsMTra1cTrkqKZzPwJQvtHwj3hr#updateTitle=true&folderId=1Sz9AyhaTenssgJXjd-IqGUZakbPORKRf&scrollTo=nLcm-ujRGzvW)
- [Dataset](https://drive.google.com/drive/folders/1sCCc7XVBMeFEnkuBeBvQ0LakaxWqbSFl?usp=drive_link)
- [Models](https://drive.google.com/drive/folders/1DNmnKNlgRKV0wcc-C7JbNCqnvPR6tDYV?usp=drive_link)

## UI Setup

1. Install `customtkinter` and `Pillow` in `pip`

2. Execute `./src/main.py`

## Attributions

> Code mostly came from [microsoft/PLOG](https://github.com/microsoft/PLOG) and [czyssrs/Logic2Text](https://github.com/czyssrs/Logic2Text)
