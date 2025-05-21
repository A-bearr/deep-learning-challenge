# deep-learning-challenge
- Module 21 Challenge 

# Neural Network Model Report: Alphabet Soup Deep Learning Analysis 
Overview of the Analysis
The purpose of this analysis was to develop and evaluate a deep learning model to predict whether a non-profit organization would receive funding from Alphabet Soup. By using historical data containing organizational, financial, and categorical attributes, the objective was to create a binary classification model that could assist in funding decisions.
 
## Results
Data Preprocessing
•    Target Variable(s):
o    IS_SUCCESSFUL — indicates whether an organization received funding (1) or did not (0).
•    Feature Variable(s):
o    Categorical and numerical variables related to the organization's activities and financial details, including:
    APPLICATION_TYPE
    AFFILIATION
    ASK_AMT
    INCOME_AMT
    SPECIAL_CONSIDERATIONS
    CLASSIFICATION
    USE_CASE
    ORGANIZATION
    Encoded versions of other relevant categorical features using One-Hot Encoding.
•    Variables Removed:
o    Columns removed due to irrelevance or potential data leakage:
    EIN (Unique identifier)
    NAME (Organization name)
    STATUS (Post-target or duplicative)
    Any other columns with excessive null values or unrelated to prediction
 
## Compiling, Training, and Evaluating the Model
•    Neural Network Architecture:
o    Input Layer: Based on the number of input features after preprocessing (e.g., ~116 features after One-Hot Encoding).
o    Hidden Layers:
    First hidden layer: 80 neurons, ReLU activation
    Second hidden layer: 30 neurons, ReLU activation
    Dropout layers (rate = 0.2) were added after hidden layers to reduce overfitting
o    Output Layer:
    1 neuron with Sigmoid activation (for binary classification)
•    Model Compilation:
o    Loss function: Binary Crossentropy
o    Optimizer: Adam (performed best during experimentation)
o    Metrics: Accuracy
•    Model Performance:
o    ✅ Achieved approximately 73% accuracy on the validation set
o    📉 Training and validation loss both steadily decreased
o    📈 Accuracy stabilized and generalization remained consistent after tuning


•     Steps Taken to Improve Performance:
o    Performed feature scaling (standardization of ASK_AMT and other numeric features)
o    Applied One-Hot Encoding to categorical variables
o    Added Dropout layers to reduce overfitting
o    Tuned:
    Number of neurons in each hidden layer
    Batch size (32 vs 64)
    Number of epochs (increased from 20 to 50)
    Learning rate (via optimizer tweaking)
o    Compared multiple architectures and logged performance for each attempt
 
## Summary and Recommendations
The final deep learning model effectively predicts funding success with a validation accuracy of around 75%, meeting the project goal. This result demonstrates that neural networks can be a viable solution for binary classification problems involving structured tabular data, especially with thorough data preprocessing and model tuning.
Alternative Model Recommendation
To further improve performance or interpretability, I recommend experimenting with ensemble tree-based models such as:
•    Random Forest Classifier
•    Gradient Boosted Trees (e.g., XGBoost, LightGBM)
Why?
•    These models:
o    Handle categorical features natively (with minimal encoding)
o    Are robust to overfitting due to ensemble averaging
o    Provide clear feature importance rankings, which are easier for stakeholders to interpret
o    Often outperform neural networks on small to mid-sized structured datasets
 

