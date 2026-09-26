# Banking Support Ticket Classifier

A machine learning project that classifies banking customer-service
messages into 77 intent categories using TF-IDF and logistic regression.

## Results

Evaluated on BANKING77's separate test set of 3,080 queries:

- Test accuracy: 88.90%
- Test macro-F1: 0.8894

## Approach

1. Checked training data for missing values and exact duplicate messages.
2. Split the 10,003 training examples into 80% training and 20% validation,
   preserving category proportions.
3. Converted text into TF-IDF features using words and two-word phrases.
4. Compared logistic regression with C=1 and C=5.
5. Selected C=5 based on validation macro-F1.
6. Retrained the pipeline on all 10,003 training examples.
7. Evaluated on the separate test set and saved the trained pipeline.

## Validation Experiments

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Logistic regression, C=1 | 84.36% | 0.8409 |
| Logistic regression, C=5 | 87.31% | 0.8745 |

## Example

Input: "My card hasn't arrived yet."

Prediction: card_arrival

## Run the Project

Open ticket_classifier.ipynb in Jupyter or VS Code with the Jupyter
extension, and run the cells from top to bottom.

Required packages: pandas, scikit-learn, matplotlib, and joblib.

An internet connection is required to download the dataset.
The notebook creates the trained model in models/ticket_classifier.joblib.

## Dataset

Uses BANKING77 by PolyAI, licensed under CC BY 4.0.

Source: https://github.com/PolyAI-LDN/task-specific-datasets

## Limitations

- Evaluated on a banking intent benchmark, not a live support system.
- Similar or ambiguous categories can be confused.
- Always predicts one of the 77 categories, even for unrelated messages.
- Does not yet include an API or user interface.

## Command-line prediction

Run `ticket_classifier.ipynb` first to train and save the model.
The generated model file is excluded from Git.

Using the Python environment where you installed the requirements:

```bash
python predict.py "My card has not arrived yet"
```

Expected output:

```text
Message: My card has not arrived yet
Predicted category: card_arrival
```