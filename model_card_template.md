# Model Card

See the Model Card paper for what it is about: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a Random-Forest-based classifier model trained to predict whether an individual's salary exceeds $50K per year based on census data. 

## Intended Use
The model can be used by organizations and researchers to analyze income disparity, understand demographic trends, and make data-driven decisions in areas such as social services, employment, and policy-making. However, tt is essential to note that the model is intended for exploratory analysis and should not be used as the sole basis for high-stakes decisions.

## Training Data
The model was trained on a publicly available dataset containing census information from the U.S. population, which includes demographic attributes such as age, education, work class, occupation, and more. The training dataset consists of 26,026 instances, with 14 features.

## Evaluation Data
The evaluation of the model was conducted on a separate test set of 6,508 instances, drawn from the same dataset. The evauation data is taken from 20% of the original data. The random forest classifier is trained with class_weight='balanced' to consider the imbalance between salaries belew $50K  and above $50k, which is 3:1. This ensures that the model's performance metrics are indicative of its generalization ability to unseen data.

## Metrics
The model was evaluated using the following metrics:
Accuracy: 0.8599
Precision: 0.7631
Recall: 0.6251
F1 Score: 0.6872
These metrics indicate that the model performs well, particularly in terms of accuracy and precision, while still maintaining a balanced recall score.

## Ethical Considerations
When using this model, it is crucial to consider the ethical implications, particularly regarding bias and fairness. The census data used for training may contain inherent biases based on historical socioeconomic conditions. Users should be aware of potential misclassifications and the impact that biased predictions can have on different demographic groups.

## Caveats and Recommendations
Data Quality: Ensure that the input data used for prediction is clean and well-structured, as data quality directly affects model performance.
Bias and Fairness: Continuously monitor the model's predictions for any biases or unfairness towards certain demographic groups. Adjustments and retraining may be necessary to mitigate these issues.
Model Updates: Periodically retrain the model with updated census data to capture changing socioeconomic conditions.
Data imbalance: When traning with new data, consider traning with varios ratio of spliting the data to training and evaluation data sets as the salary classification of the data may not be balanced. Alternatively, try try traning with K-fold split. 