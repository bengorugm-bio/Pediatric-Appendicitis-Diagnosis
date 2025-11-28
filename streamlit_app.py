import streamlit as st
import pandas as pd
import os
import numpy as np

# joblib import fallback: try top-level joblib, then sklearn's bundled joblib, then raise
try:
    import joblib
except Exception:
    try:
        # older scikit-learn versions exposed joblib under sklearn.externals
        from sklearn.externals import joblib  # type: ignore
    except Exception:
        joblib = None

st.set_page_config(page_title="Appendicitis Diagnosis Predictor", layout="centered")

st.title("Pediatric Appendicitis Diagnosis Predictor")
st.write("Enter patient features to predict appendicitis diagnosis. This app will try to load a saved model and scaler from the working directory.")

# Paths
MODEL_PATH = "random_forest_model.pkl"
RESULTS_PATH = "modeling_results.pkl"
SCALER_PATH = "scaler.pkl"
SCALER_PICKLE_PATH = "scaler_pickle.pkl"

# Containers for diagnostics
diag = st.expander("Diagnostics / loaded objects", expanded=True)

model = None
scaler = None
feature_names = None
loaded_from = None

# Helper: robust load using joblib/pandas/pickle fallbacks
def robust_load_model_file(path):
    """Try several strategies to load a model/pickle file.
    Returns loaded object or raises the last exception.
    """
    # If it's the modeling results (likely saved via pandas.to_pickle), prefer pandas.read_pickle
    if os.path.basename(path) == RESULTS_PATH:
        try:
            return pd.read_pickle(path)
        except Exception as e_pd:
            last_exc = e_pd
            # fall back to joblib if available
            if joblib is not None:
                try:
                    return joblib.load(path)
                except Exception as e_j:
                    last_exc = e_j
            # final fallback to pickle
            import pickle

            with open(path, 'rb') as f:
                try:
                    return pickle.load(f)
                except Exception as e_pickle:
                    raise last_exc
    else:
        # For model/scaler files prefer joblib, then pickle
        if joblib is not None:
            try:
                return joblib.load(path)
            except Exception as e_j:
                last_exc = e_j
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


# 1) Try to load a standalone model file
if os.path.exists(MODEL_PATH):
    try:
        model = robust_load_model_file(MODEL_PATH)
        loaded_from = MODEL_PATH
        diag.write(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        diag.write(f"Failed to load {MODEL_PATH}: {e}")

# 2) If no standalone model, try modeling_results.pkl (fallback)
if model is None:
    if os.path.exists(RESULTS_PATH):
        try:
            results = robust_load_model_file(RESULTS_PATH)
            diag.write(f"Loaded {RESULTS_PATH}; top-level keys: {list(results.keys())}")

            # Prefer 'diagnosis' target, else first
            if 'diagnosis' in results:
                target_key = 'diagnosis'
            else:
                try:
                    target_key = next(iter(results.keys()))
                except StopIteration:
                    raise KeyError('modeling_results.pkl appears empty')

            target_obj = results[target_key]
            if isinstance(target_obj, dict) and 'models' in target_obj:
                models_map = target_obj['models']
            elif isinstance(target_obj, dict):
                models_map = target_obj
            else:
                raise TypeError('Unexpected format for modeling_results')

            diag.write(f"Using target: {target_key}. Available models: {list(models_map.keys())}")

            # Prefer random_forest
            if 'random_forest' in models_map:
                model_name = 'random_forest'
            else:
                model_name = next(iter(models_map.keys()))

            model = models_map[model_name]
            loaded_from = f"{RESULTS_PATH}:{target_key}/{model_name}"
            diag.write(f"Picked model '{model_name}' from target '{target_key}'")

            # If modeling results include selected_features, use them
            if isinstance(target_obj, dict) and 'selected_features' in target_obj:
                feature_names = target_obj['selected_features']
                diag.write(f"Using selected_features from modeling results ({len(feature_names)} features)")

        except Exception as e:
            diag.write(f"Failed to load/interpret {RESULTS_PATH}: {e}")
    else:
        diag.write(f"Neither {MODEL_PATH} nor {RESULTS_PATH} found in working directory.")

# 3) Load scaler if present - prefer pickle-compatible copy if available
if os.path.exists(SCALER_PICKLE_PATH):
    try:
        scaler = robust_load_model_file(SCALER_PICKLE_PATH)
        diag.write(f"Loaded scaler from {SCALER_PICKLE_PATH}")
    except Exception as e:
        diag.write(f"Failed to load {SCALER_PICKLE_PATH}: {e}")
elif os.path.exists(SCALER_PATH):
    try:
        scaler = robust_load_model_file(SCALER_PATH)
        diag.write(f"Loaded scaler from {SCALER_PATH}")
    except Exception as e:
        diag.write(f"Failed to load {SCALER_PATH}: {e}")
else:
    diag.write(f"No scaler found (looked for {SCALER_PICKLE_PATH} and {SCALER_PATH}); raw inputs will be passed to the model")

# 4) If model has feature_names_in_, prefer that
if model is not None and feature_names is None:
    try:
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
            diag.write(f"Detected feature names from model.feature_names_in_ ({len(feature_names)} features)")
    except Exception:
        pass

# 5) If still no feature names, fall back to example set
if feature_names is None:
    feature_names = ['Age', 'WBC', 'CRP', 'RLQ_Tenderness']
    diag.write("No feature names detected; falling back to example features: Age, WBC, CRP, RLQ_Tenderness")

# Show summary
diag.markdown(f"**Model loaded from:** {loaded_from if loaded_from else 'None'}")
if model is not None:
    diag.markdown(f"**Model type:** {type(model)}")
    try:
        # small sample of model repr
        diag.text(repr(model)[:1000])
    except Exception:
        pass
else:
    diag.warning("No model loaded — add a model file to the notebook directory and reload the app.")

# Build input form dynamically based on feature_names
st.markdown("---")
st.header("Patient inputs")

input_values = {}

# Simple heuristic for field types from names
for fname in feature_names:
    lname = fname.lower()
    if any(k in lname for k in ['age', 'years']):
        input_values[fname] = st.number_input(f"{fname}", min_value=0, max_value=120, value=5)
    elif any(k in lname for k in ['wbc', 'crp', 'count', 'level', 'mg', 'mmol']):
        # float
        input_values[fname] = st.number_input(f"{fname}", value=0.0, format="%.3f")
    elif any(k in lname for k in ['tender', 'yes', 'no', 'rlq', 'pain']):
        input_values[fname] = st.selectbox(f"{fname}", options=[0, 1])
    else:
        # generic numeric input (model expects numeric array)
        input_values[fname] = st.number_input(f"{fname}", value=0.0)

st.write("")

# Predict
if st.button("Predict Diagnosis"):
    if model is None:
        st.error("No model loaded — cannot predict. See Diagnostics for details.")
    else:
        X = pd.DataFrame([input_values])
        # Ensure columns in the order feature_names
        try:
            X = X.reindex(columns=feature_names)
        except Exception:
            pass

        # Fill missing columns with zeros
        for c in feature_names:
            if c not in X.columns:
                X[c] = 0

        # Attempt scaling
        X_in = X.values
        if scaler is not None:
            try:
                X_in = scaler.transform(X)
            except Exception as e:
                diag.write(f"Scaler transform failed: {e}; passing raw inputs")
                X_in = X.values

        try:
            pred = model.predict(X_in)
            # If binary classification with probability available, show prob
            prob = None
            if hasattr(model, 'predict_proba') and getattr(pred, '__len__', lambda: 0)():
                try:
                    prob = model.predict_proba(X_in)[:, 1]
                except Exception:
                    prob = None

            st.success(f"Predicted Diagnosis: {pred[0]}")
            if prob is not None:
                st.info(f"Predicted probability (class 1): {prob[0]:.3f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Option to save the currently chosen model and scaler (if loaded from modeling_results) as standalone files
st.markdown("---")
if loaded_from and RESULTS_PATH in loaded_from:
    if st.button("Export chosen model + scaler to standalone files (random_forest_model.pkl, scaler.pkl)"):
        try:
            joblib.dump(model, MODEL_PATH)
            if scaler is not None:
                joblib.dump(scaler, SCALER_PATH)
            st.success("Exported model and scaler to working directory")
        except Exception as e:
            st.error(f"Export failed: {e}")

st.caption("Note: This app uses simple heuristics to build the input form. Modify `feature_names` mapping in the script if your model expects different inputs or encodings.")
