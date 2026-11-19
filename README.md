# GamePulse Review Sentiment with PyTorch

## Project Description
This project builds a small PyTorch neural network that classifies video game review text as positive or negative. It adapts GamePulse review data into a reproducible course project.

## Real-World Problem
Players often face many short, mixed, or difficult-to-interpret reviews when deciding whether to try a game. A sentiment classifier can help summarize broad review polarity, while individual reviews and limitations remain visible to users.

## Project Objective
Train and compare a baseline text classifier and an updated PyTorch multilayer perceptron (MLP), evaluate them on held-out reviews, and inspect example predictions.

## Dataset
The notebook uses `data/gamepulse_sentiment_validation_sample.csv`, a 1,100-row sample from GamePulse's Steam and Xbox written-review validation data. The input is review text; the target is binary positive/negative polarity. Neutral and mixed examples are excluded from this binary experiment.

Steam labels in this sample were AI-assigned and are not human verified. Xbox labels are based on the reviewer's star rating (4–5 positive, 1–2 negative); these are proxies, not independent human sentiment annotations. The sample was copied from the local GamePulse project and stripped of review IDs and platform IDs for this starter folder. Before publishing the data, confirm that redistribution is permitted by the original sources. If it is not, remove the CSV and follow the original project's data access instructions instead.

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
The test set is held apart from fitting and model selection. The notebook reports accuracy, precision, recall, F1, a confusion matrix, and examples of correct and incorrect predictions where available. Results will be available after running the notebook; no metrics are claimed here before execution.

## Fine-Tuning or Optimization
The baseline uses 32 hidden units without dropout. The updated model increases the hidden layer to 64 units and adds 30% dropout. The notebook compares their validation and test results and discusses whether this change helped; a change is not assumed to improve performance in advance.

## Example Predictions
The notebook displays at least three held-out review texts with their expected labels, model predictions, and confidence scores.

## Results and Conclusion
Complete this section after running the notebook. Discuss the held-out results, whether the updated model improved on the baseline, and what the experiment demonstrates about using a small neural network for review sentiment.

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
