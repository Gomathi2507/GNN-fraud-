**Graph Neural Network-Based Detection of Mule Accounts and Coordinated Fraud in Digital Payment Networks
A Comparative Study Against Tabular Machine Learning Baselines**

**Project Overview:**

Digital payment fraud increasingly involves interconnected transactions, mule accounts, and coordinated fraudulent networks. Traditional machine learning models generally analyze transactions independently, relying on transaction-level features without explicitly considering relationships between connected transactions.
This project investigates whether Graph Neural Networks (GNNs) can improve illicit transaction detection by leveraging both transaction features and relationships between transactions.
The Elliptic Bitcoin Transaction Dataset is used as a publicly available proxy dataset because real-world UPI transaction data is generally not publicly accessible for research. Transactions are represented as nodes in a graph, and transaction relationships are represented as edges.
A GraphSAGE-based Graph Neural Network is developed and compared against XGBoost, a traditional tabular machine learning baseline, using a consistent experimental setup.
The project evaluates both models using metrics suitable for highly imbalanced fraud detection and includes fraud network visualization and basic model explainability.

**Objectives:**

Develop a GraphSAGE-based GNN for illicit transaction detection.
Establish an XGBoost baseline using transaction-level features.
Compare graph-based learning with conventional tabular machine learning.
Evaluate model performance under class imbalance.
Analyze the influence of neighboring transactions on fraud predictions.
Visualize transaction networks and identify suspicious patterns.
Investigate whether graph structure provides additional predictive value beyond individual transaction features.

**Dataset:**

Elliptic Bitcoin Transaction Dataset

The Elliptic dataset contains Bitcoin transactions labeled as licit, illicit, or unknown, along with transaction features and information about relationships between transactions.
It is used in this project as a research proxy for studying graph-based financial crime detection.

**Dataset characteristics:**

Nodes: Bitcoin transactions.
Edges: Relationships between transactions represented in the dataset.
Features: Transaction-level features, including aggregated transaction information.
Labels: Licit, illicit, and unknown transaction categories.

**Dataset source:**

Elliptic dataset: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
Research paper: https://arxiv.org/abs/1908.02591

**Data preprocessing:**

The preprocessing pipeline will include:

Loading transaction features, transaction identifiers, labels, and edge information.
Matching transaction identifiers across feature, label, and edge files.
Handling missing or invalid feature values.
Encoding the target labels for binary classification.
Excluding unknown labels from supervised training and evaluation.
Creating a consistent training and testing protocol for both models.
Preparing the graph structure and node features for GraphSAGE.

**Methodology:**

The project compares two approaches.

1. XGBoost — Tabular Machine Learning Baseline
XGBoost predicts whether a transaction is illicit using its individual feature vector.
Input: Transaction-level features.
Output: Predicted class and illicit transaction probability.
The model does not explicitly aggregate information from neighboring transactions.

Key steps:

Prepare the tabular feature matrix.
Split labeled transactions into training and testing sets.
Train an XGBoost classifier.
Generate class predictions and probability scores.
Evaluate performance using classification and ranking metrics.

2. GraphSAGE — Graph Neural Network
GraphSAGE learns node representations by combining transaction features with information aggregated from neighboring nodes.
Input: Node features and transaction graph edges.
Output: Predicted class and illicit transaction probability for each evaluated transaction.
GraphSAGE learns representations that incorporate local graph structure, potentially helping identify suspicious patterns that are difficult to detect from individual transaction features alone.

Key steps:

Construct a graph with transactions as nodes and dataset relationships as edges.
Convert node features and labels into PyTorch tensors.
Create training, validation, and test masks.
Train a GraphSAGE model using neighborhood aggregation.
Predict illicit transaction probabilities.
Evaluate predictions using the same metrics used for XGBoost.

**Evaluation Metrics:**

Fraud detection datasets are often highly imbalanced, so accuracy alone is insufficient.

The following metrics will be used:

Precision: Proportion of transactions predicted as illicit that are actually illicit.
Recall: Proportion of actual illicit transactions correctly identified.
F1-score: Harmonic mean of precision and recall.
ROC-AUC: Measures ranking performance across different classification thresholds using the receiver operating characteristic curve.
PR-AUC: Measures precision-recall performance and is especially informative for imbalanced classification.
The comparison will focus on precision, recall, F1-score, ROC-AUC, and PR-AUC.

For PR-AUC, average precision and trapezoidal area under the precision-recall curve are different calculations. The implementation should specify which one is reported.

**References:**

1. Weber, M., et al. Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. KDD Workshop on Anomaly Detection in Finance, 2019. https://arxiv.org/abs/1908.02591
2. Hamilton, W. L., Ying, R., and Leskovec, J. Inductive Representation Learning on Large Graphs. NeurIPS, 2017. https://arxiv.org/abs/1706.02216
3. XGBoost Documentation: https://xgboost.readthedocs.io/
4. PyTorch Documentation: https://pytorch.org/docs/
5. PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/
