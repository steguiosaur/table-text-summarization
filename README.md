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

## Required

Currently working on `python 3.12.7`

Access experimentations, models, and datasets on the following links:

- [Google Colab](https://colab.research.google.com/drive/1XpQio7HnYCV1dKOzhhLsQaZoM6DDuEoz?usp=sharing)
- [Datasets](https://drive.google.com/drive/folders/1aSSZ0-xeEeNbN2v90fhNLL9ed-MQ9dBT?usp=drive_link)
- [Models](https://drive.google.com/drive/folders/1-61sJKlLrn1MpEU2jTtza6EeFL0_2hi5?usp=drive_link)
- [Results](https://drive.google.com/drive/folders/1-6FIolQblpU7ufh9QuRw7fFGqZB6Hhyh?usp=sharing)

## Software Tool Setup

1. Install dependencies using the command `pip install -r requirements.txt`

2. Download model from the Models folder and move it to `./models/` folder

3. Execute `./src/main.py` to load the software

## Attributions

> Code mostly came from [microsoft/PLOG](https://github.com/microsoft/PLOG) and [czyssrs/Logic2Text](https://github.com/czyssrs/Logic2Text)
