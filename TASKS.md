# TASKS

List of things to do and questions to answer.

## Task on UI

- Add field for Target summary
- Add R1, R2, RL calculation
- Add NP, NC, NS calculation
- UI on bottom for evaluation results

## Maam Fabregas

- Where to see the faithfulness? Based on Three metrics
- Show the calculated factuality in terms of the metrics
- How to see the calculated factuality? Include in the UI
- How to see the other models evaluation?
- How it will be checked by the expert? Show the result of other models

- How the system answers the SOP1? Show it in the UI
- How the expert will check the results?
- Put result of the evaluations on the Appendix
- Explain in less technical way, the calculation of ROUGE scores
- Tool should show the scores already and how it was interpreted
- You should have the output of every other models to compare
- Show the calculated factuality in tool for different models
- The current outputs of ROUGE is not the desired output of ROUGE
  - R1 good if >50%, R2 good if >40%
- In terms of human evaluation, how do you answer it?

- Show the NP, NC, NS in UI
- What does the result mean with respect to the score
- How did you calculate the factuality of models based on the metrics
- Did not show the calculated factuality
- You have your program and dapat nakalagay sa interface nyo
- Show metrics on the UI

- In prototype, show the calculated factuality, human eval
- What are the parameters of the human evaluation? Where is it referenced?
- SOP2 is expert
- We compare the output of our models to the outputs of other models
- Expert will show ratings.

- Metrics should have Related Literature (Chapter 2)
- Answer SOP with respect to the
- Tackle the "in terms of" when talking about the generated summary
- Show all the models in tool
- Check the output on ROUGE
- Mistake on the ROUGE computation

## Final Defense

- [ ] fix github
- [ ] fix colab
- [x] chapter 3 np nc ns
- [ ] chapter 4
- [ ] chapter 5

## Questions

1. How to use the generated model?

   - Load the model using `transformers` and `torch` package.
     See file `generation.py` for sample loading and inference.

2. What would the input look like when we deal with tabular data?
3. How will the content selection be done so that we can get the idea we want to
   collect?
4. How does the logical form being formed in this particular task?
5. How would you integrate SciGen for the training? Would you need to form its
   own logical forms or just highlight the particular information within the
   table and its identified logical classification to form logical forms?
6. What is an optimizer? Why would you use AdamW and Adafactor?
7. Identify how would you evaluate the results of this model.
8. Identify how highlighting works.
9. What is BART? Why use it?

   - BART stands for Bidirectional and Auto-Regressive Transformers which is a
     sequence-to-sequence model. See link for further understanding [BART MODEL](https://www.projectpro.io/article/transformers-bart-model-explained/553)

10. Learn how this logical form representation is related to compiler design.
11. What is intermediate meaning representations?
12. Explain why you need logical forms in this.
13. Why you picked SciGen dataset?
14. How the textual data affect the output text being generated?
15. How do you define faithfulness?
16. Why is representing tabular data to logical form important?
17. Why not create your own model instead of using pretrained models?

## Tasks

- [x] Finetune the task text on the model for CONTLOG
- [x] Try running the model generated from pretraining
- [x] Identify what would be your input or how to modify tabular data along text
- [ ] Create diagrams that would describe what you are aiming for
- [ ] Create the content selection idea you have
- [ ] Update paper

## Create these Diagrams/Tables

- [ ] Faithfulness definition

- [ ] Overall System Architecture (revise)
- [ ] Diagrams pertaining to tabular data indexing
- [ ] How textual data selects or highlights the table cells
- [ ] Content selection architecture (revise)

- [ ] Construction of Logical form grammar
- [ ] Function definition of Logical forms
- [ ] Logical form sampling tables
- [ ] Process of forming logical forms

- [ ] Model architecture
- [ ] Pretraining architecture of Table2Logic Data
- [ ] Finetuning architecture

## Ideas

1. Our proposed implementation

   - Detect salient topics per sentence segment in textual data that are
     different to one another. Could use TF-IDF?
   - Filter them and determine if there are tabular data values present in each
     sentence
   - If multiple table values are present that corresponds to a row column,
     highlight them

   ABOUT THE ASSIGNMENT OF LOGICAL OPERATIONS on PREPROCESSING

   - Using keyword triggers, assign them a logical form function to calculate
   - Using the logical form classifier, classify if what logic type is the
     sentence and assign it to them

## Parts to update on paper

- [ ] Whole methodology

## What did we do. Currently

- [x] Pretrained the `facebook/bart-large` model on table to logic dataset by
      contlog
- [x] Finetuned the model that was pretrained on `facebook/bart-large` model for
      table to text tasks
- [x] Finetuned it again for summarization task using SciGen dataset

## Some solutions

1. Check this link for fixing the issue on `pyrouge`
   - [pyrouge fix](https://stackoverflow.com/questions/45894212/installing-pyrouge-gets-error-in-ubuntu)
