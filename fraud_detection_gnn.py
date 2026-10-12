#Fraud detecion

import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected
from torch_geometric.explain import Explainer, GNNExplainer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve


# =============================================================================
# SETTINGS  (everything you might want to change is here)
# =============================================================================
DATA_FOLDER = '.'

# Time split. 0.7 of the 49 time steps = 34 train steps (the standard benchmark).
TRAIN_FRACTION = 0.7          # share of all steps used for train + validation
INNER_TRAIN_FRACTION = 0.7    # share of those steps used to fit; the rest is validation

CLIP_QUANTILE = 0.001         # extreme feature values are clipped at this quantile
SEEDS = [0, 1, 2, 3, 4]       # GraphSAGE is trained once per seed

# GraphSAGE settings (fixed defaults, not tuned)
HIDDEN_SIZE = 128
DROPOUT = 0.3
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.0005
GRAD_CLIP = 1.0
MAX_EPOCHS = 300
PATIENCE = 30                 # stop if validation score does not improve for this many epochs

# XGBoost settings (depth and learning rate fixed; number of trees found automatically)
XGB_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_MAX_TREES = 2000
XGB_PATIENCE = 50

ENSEMBLE_STEPS = 21           # how many GNN weights (0.0, 0.05, ..., 1.0) to try
EXPLAINER_EPOCHS = 100
TOP_K = 10                    # how many features / edges to print in the explanation

# From the dataset paper (Weber et al. 2019): 93 local + 72 aggregated features
N_LOCAL_FEATURES = 93
N_AGGREGATED_FEATURES = 72

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# =============================================================================
# STEP 1: LOAD THE DATA
# =============================================================================
features = pd.read_csv(DATA_FOLDER + '/elliptic_txs_features.csv', header=None)
edges = pd.read_csv(DATA_FOLDER + '/elliptic_txs_edgelist.csv')
classes = pd.read_csv(DATA_FOLDER + '/elliptic_txs_classes.csv')

# Column 0 is the transaction id, column 1 is the time step, the rest are features
n_features = features.shape[1] - 2
if n_features != N_LOCAL_FEATURES + N_AGGREGATED_FEATURES:
    raise ValueError('Unexpected number of features: ' + str(n_features))

feature_names = []
for i in range(n_features):
    feature_names.append('feat_' + str(i))
feature_names = np.array(feature_names)

features.columns = ['txId', 'time_step'] + list(feature_names)
classes.columns = ['txId', 'class']

# Attach the label to every transaction
data = features.merge(classes, on='txId', how='left')
if len(data) != len(features):
    raise ValueError('Merging labels changed the number of transactions')

# Labels: 1 = illicit, 0 = licit, -1 = unknown
label_map = {'1': 1, '2': 0, 'unknown': -1}
labels = data['class'].astype(str).map(label_map)
if labels.isna().any():
    raise ValueError('classes.csv contains labels I do not recognise')

# Give every transaction a row number 0..N-1 (the graph library needs numbers)
txid_to_index = {}
for index, txid in enumerate(data['txId']):
    txid_to_index[txid] = index

source = edges['txId1'].map(txid_to_index)
target = edges['txId2'].map(txid_to_index)
both_found = source.notna() & target.notna()
print('Edges skipped because a transaction was missing:', int((~both_found).sum()),
      'of', len(edges))

source_ids = source[both_found].astype(np.int64).values
target_ids = target[both_found].astype(np.int64).values
edge_index = torch.tensor(np.vstack([source_ids, target_ids]), dtype=torch.long)

X_all = data[list(feature_names)].values.astype(np.float32)
Y = labels.values.astype(np.int64)
T = data['time_step'].values.astype(np.int64)

print('Transactions:', len(Y), '| edges:', edge_index.size(1))
print('Illicit:', int((Y == 1).sum()), '| licit:', int((Y == 0).sum()),
      '| unknown:', int((Y == -1).sum()))

# Elliptic edges should only connect transactions from the same time step.
# If that is true, the graph cannot leak information from the future into the past.
same_step_share = (T[edge_index[0].numpy()] == T[edge_index[1].numpy()]).mean()
print('Share of edges inside one time step:', round(float(same_step_share), 4))
if same_step_share < 1.0:
    print('WARNING: some edges cross time steps, so temporal leakage is possible')

# The GNN treats every edge as two-way
edge_index_u = to_undirected(edge_index).to(device)


# =============================================================================
# STEP 2: SPLIT BY TIME  (inner-train | validation | test)
# =============================================================================
all_steps = np.sort(np.unique(T))
n_train_steps = int(round(TRAIN_FRACTION * len(all_steps)))
n_inner_steps = int(round(INNER_TRAIN_FRACTION * n_train_steps))

inner_train_steps = all_steps[:n_inner_steps]
val_steps = all_steps[n_inner_steps:n_train_steps]
test_steps = all_steps[n_train_steps:]

print('Inner-train steps:', inner_train_steps[0], 'to', inner_train_steps[-1])
print('Validation steps: ', val_steps[0], 'to', val_steps[-1])
print('Test steps:       ', test_steps[0], 'to', test_steps[-1])

is_labeled = Y != -1
inner_train_mask = is_labeled & np.isin(T, inner_train_steps)
val_mask = is_labeled & np.isin(T, val_steps)
trainval_mask = inner_train_mask | val_mask
test_mask = is_labeled & np.isin(T, test_steps)

print('Labeled inner-train:', int(inner_train_mask.sum()),
      '| illicit:', int((Y[inner_train_mask] == 1).sum()))
print('Labeled validation: ', int(val_mask.sum()),
      '| illicit:', int((Y[val_mask] == 1).sum()))
print('Labeled test:       ', int(test_mask.sum()),
      '| illicit:', int((Y[test_mask] == 1).sum()))

y_test = Y[test_mask]
labels_tensor = torch.tensor(Y, device=device)


# =============================================================================
# STEP 3: SMALL HELPER FUNCTIONS
# =============================================================================
def scale_features(X, fit_nodes):
    """Clip extreme values, then standardise. Uses ONLY the nodes in fit_nodes
    to compute the clipping limits, the mean and the standard deviation."""
    X_fit = X[fit_nodes]
    low = np.quantile(X_fit, CLIP_QUANTILE, axis=0)
    high = np.quantile(X_fit, 1 - CLIP_QUANTILE, axis=0)
    X_fit_clipped = np.clip(X_fit, low, high)
    mean = X_fit_clipped.mean(axis=0)
    std = X_fit_clipped.std(axis=0)
    std[std == 0] = 1.0                      # constant column: avoid dividing by zero
    X_scaled = (np.clip(X, low, high) - mean) / std
    return X_scaled.astype(np.float32)


def class_weight_options(y):
    """Three ways to weight the rare illicit class: none, square root, full ratio."""
    ratio = (y == 0).sum() / (y == 1).sum()
    options = [('none', 1.0),
               ('sqrt', float(np.sqrt(ratio))),
               ('full', float(ratio))]
    return options


def find_best_threshold(y_true, probs):
    """The probability cut-off that gives the best F1 on the data we pass in."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    precision = precision[:-1]               # one fewer value than thresholds
    recall = recall[:-1]
    total = precision + recall
    total[total == 0] = 1e-12                # avoid dividing by zero
    f1 = 2 * precision * recall / total
    return float(thresholds[np.argmax(f1)])


def get_metrics(y_true, probs, threshold):
    predicted = (probs >= threshold).astype(int)
    result = {}
    result['Precision'] = precision_score(y_true, predicted, zero_division=0)
    result['Recall'] = recall_score(y_true, predicted, zero_division=0)
    result['F1'] = f1_score(y_true, predicted, zero_division=0)
    result['ROC_AUC'] = roc_auc_score(y_true, probs)
    result['PR_AUC'] = average_precision_score(y_true, probs)
    result['Threshold'] = threshold
    return result


def f1_for_each_step(y_true, probs, threshold, steps):
    result = {}
    for step in np.unique(steps):
        rows = steps == step
        predicted = probs[rows] >= threshold
        result[int(step)] = f1_score(y_true[rows], predicted, zero_division=0)
    return pd.Series(result)


# ---------------------------- GraphSAGE model --------------------------------
class GraphSAGE(torch.nn.Module):
    """Three GraphSAGE layers, with batch-norm and dropout between them."""

    def __init__(self, in_size, hidden_size, out_size, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_size, hidden_size)
        self.norm1 = torch.nn.BatchNorm1d(hidden_size)
        self.conv2 = SAGEConv(hidden_size, hidden_size)
        self.norm2 = torch.nn.BatchNorm1d(hidden_size)
        self.conv3 = SAGEConv(hidden_size, out_size)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        return x


def gnn_probabilities(model, X, rows):
    """Probability of 'illicit' for the given rows."""
    model.eval()
    with torch.no_grad():
        output = model(X, edge_index_u)
        probs = F.softmax(output[rows], dim=1)[:, 1]
    return probs.cpu().numpy()


def make_gnn(n_inputs, seed):
    torch.manual_seed(seed)
    model = GraphSAGE(n_inputs, HIDDEN_SIZE, 2, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    return model, optimizer


def train_one_epoch(model, optimizer, X, train_rows, class_weights):
    model.train()
    optimizer.zero_grad()
    output = model(X, edge_index_u)
    loss = F.cross_entropy(output[train_rows], labels_tensor[train_rows],
                           weight=class_weights)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()


def train_gnn_with_validation(X, train_rows, val_rows, pos_weight, seed):
    """Train until the validation score stops improving.
    Returns the best model, the epoch where it was best, and its validation probabilities."""
    model, optimizer = make_gnn(X.size(1), seed)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float, device=device)
    y_val = labels_tensor[val_rows].cpu().numpy()

    best_score = -1.0
    best_epoch = 0
    best_probs = None
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_one_epoch(model, optimizer, X, train_rows, class_weights)

        val_probs = gnn_probabilities(model, X, val_rows)
        score = average_precision_score(y_val, val_probs)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_probs = val_probs
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_probs


def train_gnn_fixed_epochs(X, train_rows, pos_weight, seed, n_epochs):
    """Train for exactly n_epochs (used for the final model, no validation data left)."""
    model, optimizer = make_gnn(X.size(1), seed)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float, device=device)
    for epoch in range(n_epochs):
        train_one_epoch(model, optimizer, X, train_rows, class_weights)
    return model


# ------------------------------ XGBoost --------------------------------------
def train_xgb_with_validation(X_train, y_train, X_val, y_val, scale_pos_weight):
    """Add trees until the validation score stops improving."""
    model = xgb.XGBClassifier(n_estimators=XGB_MAX_TREES,
                              max_depth=XGB_DEPTH,
                              learning_rate=XGB_LEARNING_RATE,
                              scale_pos_weight=scale_pos_weight,
                              eval_metric='aucpr',
                              early_stopping_rounds=XGB_PATIENCE,
                              random_state=SEEDS[0],
                              n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_xgb_fixed_trees(X_train, y_train, scale_pos_weight, n_trees):
    model = xgb.XGBClassifier(n_estimators=n_trees,
                              max_depth=XGB_DEPTH,
                              learning_rate=XGB_LEARNING_RATE,
                              scale_pos_weight=scale_pos_weight,
                              random_state=SEEDS[0],
                              n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def rows_from_mask(mask):
    """Turn a True/False array into a tensor of row numbers."""
    return torch.tensor(np.flatnonzero(mask), dtype=torch.long, device=device)


# =============================================================================
# STEP 4: ONE FULL EXPERIMENT FOR A GIVEN SET OF FEATURE COLUMNS
# =============================================================================
def run_experiment(name, feature_columns):
    print('\n==========', name, '(' + str(len(feature_columns)) + ' features) ==========')
    X = X_all[:, feature_columns]
    chosen = {}

    # ---------------------------------------------------------------
    # Part A: choose settings using inner-train -> validation only
    # ---------------------------------------------------------------

    # --- XGBoost: try each class weight, keep the best on validation ---
    best_xgb = None
    for weight_name, weight in class_weight_options(Y[inner_train_mask]):
        model = train_xgb_with_validation(X[inner_train_mask], Y[inner_train_mask],
                                          X[val_mask], Y[val_mask], weight)
        n_trees = model.best_iteration + 1
        val_probs = model.predict_proba(X[val_mask], iteration_range=(0, n_trees))[:, 1]
        score = average_precision_score(Y[val_mask], val_probs)
        print('  XGBoost  weight =', weight_name, '| validation PR-AUC =', round(score, 4),
              '| trees =', n_trees)

        if best_xgb is None or score > best_xgb['score']:
            best_xgb = {'name': weight_name, 'weight': weight, 'score': score,
                        'trees': n_trees, 'val_probs': val_probs}

    xgb_val_probs = best_xgb['val_probs']
    chosen['xgboost'] = {'class_weight_name': best_xgb['name'],
                         'scale_pos_weight': best_xgb['weight'],
                         'trees': best_xgb['trees']}

    # --- GraphSAGE: scale features using inner-train nodes only ---
    X_scaled = scale_features(X, np.isin(T, inner_train_steps))
    X_tensor = torch.tensor(X_scaled).to(device)
    inner_train_rows = rows_from_mask(inner_train_mask)
    val_rows = rows_from_mask(val_mask)

    # try each class weight (first seed only), keep the best on validation
    best_gnn = None
    for weight_name, weight in class_weight_options(Y[inner_train_mask]):
        _, best_epoch, val_probs = train_gnn_with_validation(
            X_tensor, inner_train_rows, val_rows, weight, SEEDS[0])
        score = average_precision_score(Y[val_mask], val_probs)
        print('  GraphSAGE weight =', weight_name, '| validation PR-AUC =', round(score, 4),
              '| best epoch =', best_epoch)

        if best_gnn is None or score > best_gnn['score']:
            best_gnn = {'name': weight_name, 'weight': weight, 'score': score}

    # run every seed with the chosen weight
    all_val_probs = []
    all_best_epochs = []
    for seed in SEEDS:
        _, best_epoch, val_probs = train_gnn_with_validation(
            X_tensor, inner_train_rows, val_rows, best_gnn['weight'], seed)
        all_val_probs.append(val_probs)
        all_best_epochs.append(best_epoch)

    gnn_val_probs = np.mean(all_val_probs, axis=0)
    final_epochs = int(np.median(all_best_epochs))
    chosen['graphsage'] = {'class_weight_name': best_gnn['name'],
                           'pos_weight': best_gnn['weight'],
                           'epochs': final_epochs}

    # --- Ensemble: try GNN weights from 0 to 1, keep the best on validation ---
    best_ensemble_score = -1.0
    gnn_share = 0.0
    for candidate in np.linspace(0, 1, ENSEMBLE_STEPS):
        mixed = candidate * gnn_val_probs + (1 - candidate) * xgb_val_probs
        score = average_precision_score(Y[val_mask], mixed)
        if score > best_ensemble_score:
            best_ensemble_score = score
            gnn_share = float(candidate)
    ensemble_val_probs = gnn_share * gnn_val_probs + (1 - gnn_share) * xgb_val_probs
    chosen['ensemble_gnn_share'] = gnn_share

    # --- One threshold per model, all chosen on validation ---
    thresholds = {}
    thresholds['xgboost'] = find_best_threshold(Y[val_mask], xgb_val_probs)
    thresholds['graphsage'] = find_best_threshold(Y[val_mask], gnn_val_probs)
    thresholds['ensemble'] = find_best_threshold(Y[val_mask], ensemble_val_probs)
    chosen['thresholds'] = thresholds

    print('  Chosen settings:')
    print(json.dumps(chosen, indent=4))

    # ---------------------------------------------------------------
    # Part B: retrain on inner-train + validation, then test ONCE
    # ---------------------------------------------------------------
    xgb_final = train_xgb_fixed_trees(X[trainval_mask], Y[trainval_mask],
                                      best_xgb['weight'], best_xgb['trees'])
    xgb_test_probs = xgb_final.predict_proba(X[test_mask])[:, 1]

    trainval_steps = np.concatenate([inner_train_steps, val_steps])
    X_scaled_final = scale_features(X, np.isin(T, trainval_steps))
    X_tensor_final = torch.tensor(X_scaled_final).to(device)
    trainval_rows = rows_from_mask(trainval_mask)
    test_rows = rows_from_mask(test_mask)

    gnn_models = []
    gnn_test_probs_per_seed = []
    for seed in SEEDS:
        model = train_gnn_fixed_epochs(X_tensor_final, trainval_rows,
                                       best_gnn['weight'], seed, final_epochs)
        probs = gnn_probabilities(model, X_tensor_final, test_rows)
        gnn_models.append(model)
        gnn_test_probs_per_seed.append(probs)

    gnn_test_probs = np.mean(gnn_test_probs_per_seed, axis=0)
    ensemble_test_probs = gnn_share * gnn_test_probs + (1 - gnn_share) * xgb_test_probs

    # --- Results table ---
    per_seed_results = []
    for probs in gnn_test_probs_per_seed:
        per_seed_results.append(get_metrics(y_test, probs, thresholds['graphsage']))
    per_seed_table = pd.DataFrame(per_seed_results)

    table = pd.DataFrame({
        'XGBoost': get_metrics(y_test, xgb_test_probs, thresholds['xgboost']),
        'GraphSAGE (mean over seeds)': per_seed_table.mean(),
        'GraphSAGE (std over seeds)': per_seed_table.std(),
        'Ensemble': get_metrics(y_test, ensemble_test_probs, thresholds['ensemble']),
    }).T.round(4)

    print('\n--- TEST results:', name, '---')
    print(table)
    table.to_csv('model_comparison_' + name + '.csv')

    # --- F1 at every test time step ---
    step_table = pd.DataFrame({
        'XGBoost': f1_for_each_step(y_test, xgb_test_probs,
                                    thresholds['xgboost'], T[test_mask]),
        'GraphSAGE': f1_for_each_step(y_test, gnn_test_probs,
                                      thresholds['graphsage'], T[test_mask]),
        'Ensemble': f1_for_each_step(y_test, ensemble_test_probs,
                                     thresholds['ensemble'], T[test_mask]),
    }).round(3)
    print('\nF1 at each test time step (' + name + '):')
    print(step_table)
    step_table.to_csv('f1_per_test_step_' + name + '.csv')

    # --- Save every test prediction, so charts can be made later without retraining ---
    predictions = pd.DataFrame({
        'txId': data['txId'].values[test_mask],
        'time_step': T[test_mask],
        'true_label': y_test,
        'xgboost_prob': xgb_test_probs,
        'graphsage_prob': gnn_test_probs,
        'ensemble_prob': ensemble_test_probs,
    })
    predictions.to_csv('test_predictions_' + name + '.csv', index=False)

    with open('chosen_config_' + name + '.json', 'w') as f:
        json.dump(chosen, f, indent=4)

    # keep what the explainer needs
    return {'model': gnn_models[0],
            'X': X_tensor_final,
            'probs': gnn_test_probs_per_seed[0],
            'threshold': thresholds['graphsage'],
            'feature_columns': feature_columns}


# =============================================================================
# STEP 5: RUN BOTH EXPERIMENTS
#   1) all 165 features  (the standard benchmark)
#   2) the 93 local features only  (removes the pre-computed neighbour features,
#      so it is the fair test of whether the graph itself helps)
# =============================================================================
all_columns = np.arange(N_LOCAL_FEATURES + N_AGGREGATED_FEATURES)
local_columns = np.arange(N_LOCAL_FEATURES)

result_all = run_experiment('all_features', all_columns)
result_local = run_experiment('local_features_only', local_columns)


# =============================================================================
# STEP 6: EXPLAIN ONE PREDICTION WITH GNNExplainer
# =============================================================================
def explain_one_prediction(result):
    model = result['model']
    model.eval()

    test_node_ids = np.flatnonzero(test_mask)
    probs = result['probs']

    # illicit test nodes that the model correctly flagged
    correctly_flagged = (y_test == 1) & (probs >= result['threshold'])
    print('\nCorrectly flagged illicit test nodes:', int(correctly_flagged.sum()))
    if correctly_flagged.sum() == 0:
        print('Nothing to explain.')
        return

    # pick the one the model was most sure about
    flagged_ids = test_node_ids[correctly_flagged]
    flagged_probs = probs[correctly_flagged]
    best_one = int(np.argmax(flagged_probs))
    node = int(flagged_ids[best_one])
    print('Explaining node', node, '| time step', int(T[node]),
          '| probability', round(float(flagged_probs[best_one]), 3))

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=EXPLAINER_EPOCHS),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(mode='multiclass_classification',
                          task_level='node',
                          return_type='raw'),
    )
    explanation = explainer(result['X'], edge_index_u, index=node)

    # most important features
    importance = explanation.node_mask[node].detach().cpu().numpy()
    columns = result['feature_columns']
    print('\nTop', TOP_K, 'features for this prediction:')
    order = np.argsort(importance)[::-1]
    for position in order[:TOP_K]:
        column = columns[position]
        if column < N_LOCAL_FEATURES:
            kind = 'local'
        else:
            kind = 'aggregated'
        print('  ', feature_names[column], '(' + kind + ')', round(float(importance[position]), 4))

    # most important edges (edges far from the node get a weight of exactly 0)
    edge_weights = explanation.edge_mask.detach().cpu()
    n_used = int((edge_weights > 0).sum())
    print('\nTop influential edges (', n_used, 'edges are near this node ):')
    n_show = min(TOP_K, n_used)
    top_edges = edge_weights.topk(n_show).indices
    for e in top_edges:
        start = edge_index_u[0, e].item()
        end = edge_index_u[1, e].item()
        print('  ', start, '->', end, '| weight', round(float(edge_weights[e]), 3))

    try:
        explanation.visualize_graph(path='explanation_subgraph.png')
        print('\nSaved explanation_subgraph.png')
    except Exception as error:
        print('Graph picture skipped:', error)


explain_one_prediction(result_all)