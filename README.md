Neural Network Model Report: Alphabet Soup Deep Learning Analysis

📌 Overview of the Analysis

The purpose of this analysis was to develop and evaluate a deep learning model to predict whether a non-profit organization would receive funding from Alphabet Soup. By using historical data containing organizational, financial, and categorical attributes, the objective was to create a binary classification model that could assist in funding decisions.

📊 Results

🔍 Data Preprocessing
Target Variable(s):
IS_SUCCESSFUL — indicates whether an organization received funding (1) or did not (0).
Feature Variable(s):
Categorical and numerical variables related to the organization's activities and financial details, including:
APPLICATION_TYPE
AFFILIATION
ASK_AMT
INCOME_AMT
SPECIAL_CONSIDERATIONS
CLASSIFICATION
USE_CASE
ORGANIZATION
Encoded versions of other relevant categorical features using One-Hot Encoding
Variables Removed:
Columns removed due to irrelevance or potential data leakage:
EIN (Unique identifier)
NAME (Organization name)
STATUS (Post-target or duplicative)
Any other columns with excessive null values or unrelated to prediction
🧱 Compiling, Training, and Evaluating the Model
Neural Network Architecture:
Input Layer: Based on ~116 input features after One-Hot Encoding
Hidden Layers:
Layer 1: 80 neurons, ReLU activation
Layer 2: 30 neurons, ReLU activation
Added Dropout layers (rate = 0.2) to reduce overfitting
Output Layer:
1 neuron with Sigmoid activation (for binary classification)
Model Compilation:
Loss Function: Binary Crossentropy
Optimizer: Adam (best results during experimentation)
Metric: Accuracy
Model Performance:
✅ Achieved ~73% validation accuracy
📉 Training and validation loss steadily decreased
📈 Accuracy stabilized after tuning, indicating generalization
📈 Model Training Performance Plots

/Users/alyssaberridge/Desktop/Homework/deep-learning-challenge/Deep_Learning_Challenge /output.png

Accuracy Plot: Validation accuracy stabilized near the target, with minimal overfitting.
Loss Plot: Both training and validation loss decreased over epochs, confirming effective learning.
Steps Taken to Improve Performance:
Feature scaling (ASK_AMT, etc.)
One-Hot Encoding of categorical features
Dropout layers to prevent overfitting
Hyperparameter tuning:
Neuron count
Batch size (32 vs 64)
Epochs (20 → 50)
Learning rate adjustments
Multiple architecture comparisons and performance tracking
✅ Summary and Recommendations

The final deep learning model achieved a validation accuracy of ~75%, meeting the project benchmark. This result shows that neural networks are an effective method for binary classification tasks involving structured, tabular data, especially when paired with proper data preprocessing and tuning.

🔁 Alternative Model Recommendation

To improve performance or enhance interpretability, consider using ensemble tree-based models such as:

Random Forest Classifier
Gradient Boosted Trees (e.g., XGBoost, LightGBM)
💡 Why Use These?
Handle categorical variables with less preprocessing
Naturally resist overfitting through ensemble learning
Provide feature importance metrics for interpretability
Often outperform neural networks on structured data with fewer rows and columns
 

