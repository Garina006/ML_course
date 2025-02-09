import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Any, Union


def split_data(df: pd.DataFrame, target_col: str) -> Dict[str, pd.DataFrame]:
    """
    Splits the dataset into training and validation sets while preserving the class distribution.

    Args:
        df (pd.DataFrame): The original dataset containing features and target column.
        target_col (str): The name of the target column used for stratification.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing:
            - 'train' (pd.DataFrame): Training set (80% of data).
            - 'val' (pd.DataFrame): Validation set (20% of data).
    """
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    return {'train': train_df, 'val': val_df}


def create_inputs_targets(df_dict: Dict[str, pd.DataFrame], input_cols: list, target_col: str) -> Dict[str, Any]:
    """
    Create inputs and targets for training, validation, and test sets.

    Args:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing the train, validation, and test dataframes.
        input_cols (list): List of input columns.
        target_col (str): Target column.

    Returns:
        Dict[str, Any]: Dictionary containing inputs and targets for train, val, and test sets.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data


def scale_numeric_features(data: Dict[str, Any], numeric_cols: list) -> None:
    """
    Scale numeric features using MinMaxScaler.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train, val, and test sets.
        numeric_cols (list): List of numerical columns.
    """
    scaler = MinMaxScaler().fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = scaler.transform(data[f'{split}_inputs'][numeric_cols])
    return scaler


def encode_categorical_features(data: Dict[str, Dict[str, pd.DataFrame]], categorical_cols: List[str]) -> None:
    """
    Encode categorical features using Label Encoding for binary features
    and Target Encoding for features with more than 2 unique values.

    The function modifies the `data` dictionary in place by:
    - Creating encoded features with prefix "Encoded_".
    - Removing the original categorical columns from both `train_inputs` and `val_inputs`.

    Args:
        data (Dict[str, Dict[str, pd.DataFrame]]):
            A dictionary containing train and validation datasets. It must have:
            - 'train_inputs': DataFrame with training input features.
            - 'train_targets': Series or DataFrame with target values.
            - 'val_inputs': DataFrame with validation input features.
        categorical_cols (List[str]):
            A list of categorical column names that need to be encoded.

    Returns:
        None: The function modifies `data` in place.
    """
    encoders = {}

    for col in categorical_cols:
        num_categories = data['train_inputs'][col].nunique()

        if num_categories == 2:
            le = LabelEncoder()
            le_encoder = le.fit(data['train_inputs'][col])
            encoders[col] = le_encoder
        else:
            te = TargetEncoder()
            te_encoder = te.fit(data['train_inputs'][[col]], data['train_targets'])
            encoders[col] = te_encoder

    for split in ['train', 'val']:
        encoded_cols = []

        for col in categorical_cols:
            encoder = encoders[col]
            data[f'{split}_inputs'][f"Encoded_{col}"] = encoder.transform(data[f'{split}_inputs'][[col]])
            encoded_cols.append(f"Encoded_{col}")

        data[f'{split}_inputs'].drop(columns=categorical_cols, inplace=True)
    return encoded_cols, encoders


def preprocess_data(raw_df: pd.DataFrame, input_cols, target_col, scaler_numeric: bool = True) -> Dict[str, Any]:
    """
    Preprocesses the raw dataframe by:
    - Splitting into train and validation sets.
    - Scaling numeric features (if enabled).
    - Encoding categorical features.
    - Returning processed data and fitted transformers.

    Args:
        raw_df (pd.DataFrame): The raw dataset.
        input_cols (List[str]): List of feature column names.
        target_col (str): Name of the target column.
        scaler_numeric (bool, optional): Whether to apply numerical feature scaling. Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'train_X' (pd.DataFrame): Processed training input features.
            - 'train_y' (pd.Series or pd.DataFrame): Training target values.
            - 'val_X' (pd.DataFrame): Processed validation input features.
            - 'val_y' (pd.Series or pd.DataFrame): Validation target values.
            - 'scaler' (MinMaxScaler or None): Fitted scaler for numerical features, if `scaler_numeric=True`.
            - 'encoders' (Dict[str, Any]): Dictionary of fitted encoders for categorical features.
    """

    split_dfs = split_data(raw_df, target_col)
    data = create_inputs_targets(split_dfs, input_cols, target_col)

    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes('object').columns.tolist()

    if scaler_numeric:
        scaler = scale_numeric_features(data, numeric_cols)
    encoded_cols, encoders = encode_categorical_features(data, categorical_cols)

    # Extract X_train, X_val, X_test
    X_train = data['train_inputs'][numeric_cols + encoded_cols]
    X_val = data['val_inputs'][numeric_cols + encoded_cols]

    return {
        'train_X': X_train,
        'train_y': data['train_targets'],
        'val_X': X_val,
        'val_y': data['val_targets'],
        'scaler': scaler,
        'encoders': encoders

    }


def preprocess_new_data(new_df: pd.DataFrame, scaler, encoders: Dict[str, Any],
                        input_cols: List[str], scaler_numeric: bool = True) -> pd.DataFrame:
    """
     Preprocesses new unseen data using the previously fitted scaler and encoders.

     Args:
         new_df (pd.DataFrame): The new dataset that needs preprocessing.
         scaler (MinMaxScaler): Fitted MinMaxScaler from preprocess_data().
         scaler_numeric (bool, optional): Whether to apply numerical feature scaling. Defaults to True.
         encoders (Dict[str, Any]): Dictionary of fitted encoders from preprocess_data().
         input_cols (List[str]): List of input feature column names.

     Returns:
         pd.DataFrame: Processed DataFrame ready for model prediction.
     """


    test_inputs = new_df[input_cols]
    numeric_cols = test_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = test_inputs.select_dtypes('object').columns.tolist()

    if scaler_numeric:
        test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

    encoded_cols = []
    for col in categorical_cols:
        encoder = encoders[col]
        test_inputs[f"Encoded_{col}"] = encoder.transform(test_inputs[[col]])
        encoded_cols.append(f"Encoded_{col}")

    test_inputs.drop(columns=categorical_cols, inplace=True)

    X_test = test_inputs[numeric_cols + encoded_cols]

    return X_test


def compute_auroc_and_build_roc(model: BaseEstimator, inputs: np.ndarray, targets: np.ndarray, name: str = '') -> float:
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (AUROC) 
    and plots the ROC curve for a given trained model.

    Args:
        model (BaseEstimator): A trained classification model that supports `predict_proba()`.
        inputs (np.ndarray): Feature matrix for prediction (shape: [n_samples, n_features]).
        targets (np.ndarray): True binary labels (ground truth) (shape: [n_samples]).
        name (str, optional): A name for the plot and printed AUROC score. Defaults to an empty string.

    Returns:
        float: The computed AUROC score, representing the area under the ROC curve.
    """
    
    y_pred_proba = model.predict_proba(inputs)[:, 1]

    fpr, tpr, thresholds = roc_curve(targets, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print(f'AUROC for {name}: {roc_auc:.2f}')

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {name}')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc
