import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

@st.cache_resource
def load_artifacts():
    with open("models/car_price_model.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return artifacts

@st.cache_data
def load_train_data():
    url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
    df = pd.read_csv(url)
    return df

artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]

df_train = load_train_data()
numeric_cols_global = df_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
target_col = "selling_price"
num_features = [c for c in numeric_cols_global if c != target_col]

df_train = load_train_data()

st.set_page_config(page_title="Car Price Regression", layout="wide")
st.title("app.py for car selling price prediction")

st.sidebar.header("menu")
page = st.sidebar.radio(
    "Choose an option:",
    ["EDA", "Predictions", "Model weights"],
)


if page == "EDA":
    st.header(" EDA")

    st.subheader("General")
    st.write("df_train shape:", df_train.shape)
    st.dataframe(df_train.head())

    st.markdown("---")

    numeric_cols = df_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    target_col = "selling_price"

    st.subheader("Numeric features hist")

    col_hist = st.selectbox("Choice of features", numeric_cols, index=numeric_cols.index(target_col))
    bins = st.slider("Bin number", min_value=10, max_value=100, value=30, step=5)

    fig, ax = plt.subplots()
    ax.hist(df_train[col_hist], bins=bins)
    ax.set_title(f"Hist: {col_hist}")
    ax.set_xlabel(col_hist)
    ax.set_ylabel("frequency / appearence")
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Connection with selling_price")

    feat = st.selectbox(
        "X-axes features",
        [c for c in numeric_cols if c != target_col]
    )

    fig2, ax2 = plt.subplots()
    ax2.scatter(df_train[feat], df_train[target_col], alpha=0.3)
    ax2.set_xlabel(feat)
    ax2.set_ylabel(target_col)
    ax2.set_title(f"{feat} vs {target_col}")
    st.pyplot(fig2)


elif page == "Predictions":
    st.header("Predictions")

    st.write(
        "Model trained on features:\n\n"
        f"**{', '.join(num_features)}**.\n\n"
        "upload CSV file ",
        "type features manually (one car)"
    )

    mode = st.radio(
        "Set predictions with",
        ["upload CSV file ", "type features manually (one car)"]
    )


    if mode == "upload CSV file ":
        uploaded_file = st.file_uploader(
            "upload csv with features columns " + ", ".join(num_features),
            type=["csv"]
        )

        if uploaded_file is not None:
            df_new = pd.read_csv(uploaded_file)
            st.subheader("subheader rows")
            st.dataframe(df_new.head())

            missing = [c for c in num_features if c not in df_new.columns]
            if missing:
                st.error(f"missing columns: {missing}")
            else:
                X_new = df_new[num_features].copy()


                X_new_scaled = scaler.transform(X_new)
                preds = model.predict(X_new_scaled)

                df_result = df_new.copy()
                df_result["predicted_price"] = preds.astype(int)

                st.subheader("Predictions df_result")
                st.dataframe(df_result)

                csv_bytes = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results in CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv"
                )


    else:
        st.subheader("type features manually (one car)")

        inputs = {}
        #
        for feat in num_features:
            if "year" in feat:
                val = st.number_input(feat, min_value=1990, max_value=2025, value=2015, step=1)
            elif "km" in feat:
                val = st.number_input(feat, min_value=0, max_value=500_000, value=60_000, step=1_000)
            elif "mileage" in feat:
                val = st.number_input(feat, min_value=0.0, max_value=50.0, value=18.0, step=0.1)
            elif "engine" in feat:
                val = st.number_input(feat, min_value=600, max_value=6000, value=1200, step=100)
            elif "power" in feat:
                val = st.number_input(feat, min_value=30, max_value=500, value=80, step=5)
            elif "seat" in feat:
                val = st.number_input(feat, min_value=2, max_value=10, value=5, step=1)
            else:
                val = st.number_input(feat, value=0.0)
            inputs[feat] = val

        if st.button("Calculate predicted selling_price"):
            X_one = pd.DataFrame([inputs])
            X_one_scaled = scaler.transform(X_one[num_features])
            pred = model.predict(X_one_scaled)[0]
            st.success(
                f"predicted selling_price: **{int(pred):,}** "
                .replace(",", " ")
            )


elif page == "Model weights":
    st.header("beta weights for model")

    coefs = model.coef_

    if len(num_features) == len(coefs):
        feature_names = list(num_features)
    elif len(num_features) > len(coefs):

        feature_names = list(num_features)[:len(coefs)]
    else:

        feature_names = list(num_features) + [
            f"feature_{i}" for i in range(len(num_features), len(coefs))
        ]

    st.write("feature_names:", feature_names, "len =", len(feature_names))
    st.write("len(coefs) =", len(coefs))

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs
    })
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df_sorted = coef_df.sort_values("abs_coef", ascending=False)

    st.subheader("beta weights df")
    st.dataframe(coef_df_sorted)

    st.subheader("barchart of abs_coef")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(coef_df_sorted["feature"], coef_df_sorted["coef"])
    ax.set_xlabel("coefficient values (beta)")
    ax.set_ylabel("Feature")
    ax.set_title("Model weights")
    plt.gca().invert_yaxis()
    st.pyplot(fig)

