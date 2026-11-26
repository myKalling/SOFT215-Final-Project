# GamePulse Review Sentiment with PyTorch

## Project Description
This project builds a small PyTorch neural network that classifies video game review text as positive or negative. It adapts GamePulse review data into a reproducible course project.

## Real-World Problem
Players often face many short, mixed, or difficult-to-interpret reviews when deciding whether to try a game. A sentiment classifier can help summarize broad review polarity, while individual reviews and limitations remain visible to users.

## Project Objective
Train and compare a baseline text classifier and an updated PyTorch multilayer perceptron (MLP), evaluate them on held-out reviews, and inspect example predictions.

## Dataset
The notebook uses `data/gamepulse_sentiment_validation_sample.csv`, a 1,100-row sample from GamePulse's Steam and Xbox written-review validation data. The input is review text; the target is binary positive/negative polarity. Neutral and mixed examples are excluded from this binary experiment.

Steam labels in this sample were AI-assigned and are not human verified. Xbox labels are based on the reviewer's star rating (4–5 positive, 1–2 negative); these are proxies, not independent human sentiment annotations. The sample was copied from the local GamePulse project and stripped of review IDs and platform IDs for this starter folder. The reviews were obtained from public websites or through API access using API keys; public availability or API access does not by itself grant permission to redistribute them. Before publishing the data, confirm that redistribution is permitted by the original sources. If it is not, remove the CSV and follow the original project's data access instructions instead.

## PyTorch Neural Network Architecture
Text is converted to TF-IDF unigram and bigram features, capped at 3,000 terms. A PyTorch MLP maps the input through a 32-unit fully connected hidden layer with ReLU to one logit, trained with `BCEWithLogitsLoss` and Adam. The tuned version tests 64 hidden units with 30% dropout. The notebook explains these choices and compares both models using the same stratified train/validation/test split.

## Installation Instructions
Use Python 3.10 or newer, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Opening the Jupyter Notebook
From this folder, run:

```bash
jupyter notebook
```

Open `GamePulse_Sentiment_PyTorch.ipynb`.

## Running the Project
Restart the notebook kernel and run every cell from top to bottom. The notebook reads the included CSV, trains both models, prints held-out metrics and sample predictions, and creates evaluation charts. Training outputs are intentionally not invented or prefilled; they appear when the notebook is run.

## Training Process
The notebook uses a fixed random seed, 68% training, 12% validation, and 20% test split, stratified by label. TF-IDF is fit on training text only. Both models use mini-batches of 32, 15 epochs, learning rate 0.001, Adam, and binary cross-entropy with logits. The best validation-loss epoch is retained for each model.

## Model Evaluation
Both models were evaluated on the same held-out test set of 157 reviews, which was not used for training or model selection:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Original: 32 hidden units, no dropout | 0.790 | 0.781 | 0.891 | 0.832 |
| Updated: 64 hidden units, 30% dropout | 0.803 | 0.796 | 0.891 | 0.841 |

The updated model had slightly higher accuracy, precision, and F1, with the same recall. The notebook also displays class-level precision, recall, and F1, a confusion matrix for the updated model, and example predictions. Because this test set is small and the labels are noisy or proxy-based, these scores should be interpreted cautiously.

## Fine-Tuning or Optimization
The baseline uses 32 hidden units without dropout. The updated model increases the hidden layer to 64 units and adds 30% dropout.

On the held-out test set, the baseline achieved 0.790 accuracy, 0.781 precision, 0.891 recall, and 0.832 F1. The updated model achieved 0.803 accuracy, 0.796 precision, 0.891 recall, and 0.841 F1. The updated model therefore improved accuracy by 0.013 and F1 by 0.009, improved precision by 0.015, and left recall unchanged. These modest differences on this small, noisy sample do not establish that the tuned model will generalize better to other reviews.

## Example Predictions
The following are examples from the updated model's held-out test predictions. The review text is abbreviated; the notebook displays the full examples and their estimated positive probabilities.

| Review excerpt | Expected label | Prediction |
|---|---|---|
| "This hopefully marks a clear intent to remake all DQ upto 8 ..." | Positive | Negative |
| "its really fun and plus its free" | Positive | Positive |
| "Usually simple coop games like this offers only less than 10 hours playtime ..." | Positive | Positive |

The first example was misclassified; the other two were correctly identified as positive.

## Results and Conclusion
The original model achieved 79.0% held-out test accuracy, 78.1% precision, 89.1% recall, and 0.832 F1. It correctly classified 124 of the 157 test reviews. Its strongest result was identifying positive reviews, with 89.1% recall and 0.83 F1 for the positive class.

The updated model achieved 80.3% held-out test accuracy, 79.6% precision, 89.1% recall, and 0.841 F1. It correctly classified 126 of the 157 test reviews. Compared with the original model, it improved test accuracy by 1.3 percentage points and F1 by 0.009 while maintaining the same positive-class recall.

The 64-unit MLP with 30% dropout outperformed the 32-unit baseline on this test sample. This suggests that the added capacity and regularization were helpful, although the improvement is small and should not be interpreted as proof of general superiority. The experiment demonstrates that a small PyTorch MLP can learn a useful binary sentiment signal from TF-IDF features, but its results remain limited by the small sample and noisy, proxy-based labels.

## Ethical Considerations
The sample's labels have different levels of reliability: Steam labels were AI-assigned and Xbox ratings are only proxies for written sentiment. The dataset may underrepresent languages, player communities, and writing styles. A wrong sentiment prediction could mischaracterize a game or its reviewers. Treat model output as a rough summary, not a substitute for reading reviews or a basis for decisions about individual players.

## Societal Impact
A benefit is helping players scan large review collections more quickly. A risk is that inaccurate or biased summaries could distort a game's reputation or obscure minority opinions, especially when the training labels are noisy.

## Known Limitations
This is a small, binary experiment; neutral and mixed reviews are excluded. Labels are proxy/AI-assigned rather than consistently human-annotated. Random review splits can place reviews of the same game in both train and test sets, so the scores do not show generalization to entirely unseen games. TF-IDF does not understand context, sarcasm, or nuanced game-specific meaning. Run results may vary slightly with library versions or hardware.

## Future Improvements
Use independently human-annotated labels, include neutral and mixed classes, split by game to test unseen-title generalization, compare against a non-neural baseline, and assess performance across platforms and genres.

## Credits and References
GamePulse project data pipeline and sentiment validation files; PyTorch documentation; scikit-learn documentation for TF-IDF and stratified splitting; pandas and Matplotlib documentation. See the original GamePulse AI Development README for upstream data collection details.
