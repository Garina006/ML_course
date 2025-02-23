import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, PolynomialFeatures, OrdinalEncoder
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple, Any


def split_data(df: pd.DataFrame, target_col: str) -> Dict[str, pd.DataFrame]:
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    return {'train': train_df, 'val': val_df}


def create_inputs_targets(df_dict: Dict[str, pd.DataFrame], input_cols: list, target_col: str) -> Dict[str, Any]:
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data


def scale_numeric_features(data: Dict[str, Any], numeric_cols: list) -> MinMaxScaler:
    scaler = MinMaxScaler().fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = scaler.transform(data[f'{split}_inputs'][numeric_cols])
    return scaler


def encode_categorical_features(data: Dict[str, Any], categorical_cols: List[str]) -> Dict[str, Any]:
    encoders = {}
    for col in categorical_cols:
        if data['train_inputs'][col].nunique() == 2:
            encoder = LabelEncoder().fit(data['train_inputs'][col])
        else:
            encoder = OrdinalEncoder().fit(data['train_inputs'][[col]])
        encoders[col] = encoder

    for split in ['train', 'val']:
        for col in categorical_cols:
            data[f'{split}_inputs'][f"Encoded_{col}"] = encoders[col].transform(data[f'{split}_inputs'][[col]])
        data[f'{split}_inputs'].drop(columns=categorical_cols, inplace=True)

    return encoders


def generate_polynomial_features(data: Dict[str, Any], numeric_cols: list, degree: int = 2) -> PolynomialFeatures:
    """
    Generate polynomial features of given degree.

    Args:
        data (Dict[str, Any]): Dictionary containing input data.
        numeric_cols (list): List of numerical feature names.
        degree (int): Degree of polynomial features (default=5).

    Returns:
        PolynomialFeatures: Fitted transformer for future use.
    """
    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
    train_poly = poly.fit_transform(data['train_inputs'][numeric_cols])
    val_poly = poly.transform(data['val_inputs'][numeric_cols])

    poly_feature_names = poly.get_feature_names_out(numeric_cols)
    data['train_inputs'] = pd.DataFrame(train_poly, columns=poly_feature_names, index=data['train_inputs'].index)
    data['val_inputs'] = pd.DataFrame(val_poly, columns=poly_feature_names, index=data['val_inputs'].index)

    return poly


def preprocess_data(raw_df: pd.DataFrame, input_cols: List[str], target_col: str, scaler_numeric: bool = True, poli: bool = True) -> Dict[str, Any]:
    """
    Preprocess raw dataset:
    - Splits into train/validation sets.
    - Applies polynomial features if enabled.
    - Scales numeric features if enabled.
    - Encodes categorical features.

    Args:
        raw_df (pd.DataFrame): The dataset.
        input_cols (List[str]): List of feature column names.
        target_col (str): Target column name.
        scaler_numeric (bool, optional): Scale numeric features (default=True).
        poli (bool, optional): Apply polynomial features (default=True).

    Returns:
        Dict[str, Any]: Dictionary containing processed data and transformers.
    """
    split_dfs = split_data(raw_df, target_col)
    data = create_inputs_targets(split_dfs, input_cols, target_col)

    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes(include='object').columns.tolist()

    scaler = None
    if scaler_numeric:
        scaler = scale_numeric_features(data, numeric_cols)

    encoders = encode_categorical_features(data, categorical_cols)
    
    poly_transformer = None
    if poli:
        poly_transformer = generate_polynomial_features(data, numeric_cols, degree=5)
        numeric_cols = list(data['train_inputs'].columns)

    
    return {
        'train_X': data['train_inputs'],
        'train_y': data['train_targets'],
        'val_X': data['val_inputs'],
        'val_y': data['val_targets'],
        'scaler': scaler,
        'encoders': encoders,
        'poly_transformer': poly_transformer
    }


def preprocess_new_data(new_df: pd.DataFrame, scaler: MinMaxScaler, encoders: Dict[str, Any],
                        poly_transformer: PolynomialFeatures, input_cols: List[str], scaler_numeric: bool = True, poli: bool = True) -> pd.DataFrame:
    """
    Preprocess new data using fitted transformers.

    Args:
        new_df (pd.DataFrame): The new dataset.
        scaler (MinMaxScaler): Fitted scaler.
        encoders (Dict[str, Any]): Dictionary of fitted encoders.
        poly_transformer (PolynomialFeatures): Fitted polynomial transformer.
        input_cols (List[str]): List of input feature columns.
        scaler_numeric (bool, optional): Scale numeric features (default=True).
        poli (bool, optional): Apply polynomial features (default=True).

    Returns:
        pd.DataFrame: Processed DataFrame ready for model prediction.
    """
    test_inputs = new_df[input_cols].copy()
    numeric_cols = test_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = test_inputs.select_dtypes(include='object').columns.tolist()

    
    if scaler_numeric:
        test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

    for col in categorical_cols:
        test_inputs[f"Encoded_{col}"] = encoders[col].transform(test_inputs[[col]])

    test_inputs.drop(columns=categorical_cols, inplace=True)
    
    if poli:
        test_poly = poly_transformer.transform(test_inputs[numeric_cols])
        poly_feature_names = poly_transformer.get_feature_names_out(numeric_cols)
        test_inputs = pd.DataFrame(test_poly, columns=poly_feature_names, index=test_inputs.index)
        numeric_cols = list(test_inputs.columns)
    
    return test_inputs


def compute_auroc_and_build_roc(model: BaseEstimator, inputs: np.ndarray, targets: np.ndarray, name: str = '') -> float:
    y_pred_proba = model.predict_proba(inputs)[:, 1]
    fpr, tpr, thresholds = roc_curve(targets, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print(f'AUROC for {name}: {roc_auc:.2f}')
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {name}')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc


def split_train_val(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into training and validation sets.

    Args:
        df (pd.DataFrame): The raw dataframe.
        target_col (str): The target column for stratification.
        test_size (float): The proportion of the dataset to include in the validation split.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation dataframes.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
    return train_df, val_df


def separate_inputs_targets(df: pd.DataFrame, input_cols: list, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate inputs and targets from the dataframe.

    Args:
        df (pd.DataFrame): The dataframe.
        input_cols (list): List of input columns.
        target_col (str): Target column.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: DataFrame of inputs and Series of targets.
    """
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    return inputs, targets
