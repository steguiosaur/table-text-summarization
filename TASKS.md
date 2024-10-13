# TASKS

List of things to do and questions to answer.

## Questions

1. How to use the generated model?
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
10. Learn how this logical form representation is related to compiler design.
11. What is intermediate meaning representations?
12. Explain why you need logical forms in this.
13. Why you picked SciGen dataset?
14. How the textual data affect the output text being generated?
15. How do you define faithfulness?
16. Why is representing tabular data to logical form important?
17. Why not create your own model instead of using pretrained models?

## Tasks

- [ ] Divide the text pretrain to half for CONTLOG training
- [ ] Finetune the task text on the model for CONTLOG
- [ ] Try running the model generated from pretraining
- [ ] Identify what would be your input or how to modify tabular data along text
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
