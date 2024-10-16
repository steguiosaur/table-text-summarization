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

This sample is from the SciGen dataset which we would be taking as an example
for evaluating the results of Table2Logic. This is not the dataset we would be
training for Table2Logic. Collected from the paper named
"_IIIDYT at IEST 2018: Implicit Emotion Classification With Deep Contextualized
Word Representations_" on "_Table 3: Classification Report (Test Set)_".

**SAMPLE INPUT**

Textual data: "lesser F1 score anger and surprise"

Tabular data:

<center>

| Emotion  | Precision | Recall | F1-score |
| -------- | --------- | ------ | -------- |
| anger    | 0.643     | 0.601  | 0.621    |
| disgust  | 0.703     | 0.661  | 0.682    |
| fear     | 0.742     | 0.721  | 0.732    |
| joy      | 0.762     | 0.805  | 0.783    |
| sad      | 0.685     | 0.661  | 0.673    |
| surprise | 0.627     | 0.705  | 0.663    |
| Average  | 0.695     | 0.695  | 0.694    |

</center>

Highlighted cells:

```sh
{("anger", i, j), ("surpise", i, j), (0.621, i, j), (0.663, i, j)}
```

**TARGET LOGICAL FORM**:

```sh
less { hop { filter_eq { all_rows ; Emotion ; anger }; F1-score};
    hop { filter_eq {all_rows ; Emotion ; surprise }; F1-score}} = true
```

**TARGET TEXT** (based on LF): "Anger has a lower F1-Score than the emotion surprise"

## Challenges

- Training requires huge computational power

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

## Attributions

> Code mostly came from [microsoft/PLOG](https://github.com/microsoft/PLOG) and [czyssrs/Logic2Text](https://github.com/czyssrs/Logic2Text)
