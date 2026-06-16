from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
CODE_DIR = PROJECT_DIR / "code"
MODEL_PATH = PROJECT_DIR / "models" / "final_model_bundle.joblib"
METADATA_PATH = PROJECT_DIR / "models" / "final_model_metadata.json"
TABLE_PATH = PROJECT_DIR / "processed" / "publication_tables.xlsx"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def install_sklearn_pickle_compat() -> None:
    try:
        import sklearn.compose._column_transformer as column_transformer
    except Exception:
        return
    if not hasattr(column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        column_transformer._RemainderColsList = _RemainderColsList


STAGE_ORDER = {
    "pre_infusion": 0,
    "post_infusion": 1,
    "combined": 2,
    "unknown": 3,
}


VALUE_LABELS = {
    "prior_hct": {"0": "No prior HCT", "1": "Prior HCT"},
    "cart_product": {"1": "Kymriah", "2": "Yescarta"},
    "treatment_background": {"0": "No bridging therapy", "1": "Received bridging therapy"},
    "age_group_disease_specific": {
        "1": "Age <29 years",
        "2": "Age 30-39 years",
        "3": "Age 40-49 years",
        "4": "Age 50-59 years",
        "5": "Age 60-69 years",
        "6": "Age >=70 years",
    },
    "race": {
        "1": "White",
        "2": "Black or African American",
        "3": "Asian",
        "4": "Native Hawaiian or other Pacific Islander",
        "5": "American Indian or Alaska Native",
        "7": "Other race",
        "8": "More than one race",
    },
    "performance_status": {"1": "KPS 90-100%", "2": "KPS 80-89%", "3": "KPS <80%"},
    "comorbidity": {"0": "HCT-CI 0", "1": "HCT-CI 1-2", "2": "HCT-CI 3-4", "3": "HCT-CI >=5"},
    "pre_cart_infection_history": {"0": "No pre-CAR-T infection history", "1": "Pre-CAR-T infection history"},
    "lymphoma_transformation": {"0": "Transformed lymphoma", "1": "De novo lymphoma"},
    "disease_status_pre_cart": {
        "1": "Complete response (CR)",
        "2": "Partial response (PR)",
        "3": "Resistant disease",
        "4": "Relapse, untreated",
        "5": "Untreated disease",
        "8": "Unknown disease status",
    },
    "prior_treatment_lines": {"1": "1 prior therapy line", "2": "2 prior therapy lines", "3": "3 prior therapy lines", "4": ">=4 prior therapy lines"},
    "disease_subtype": {"1": "Follicular lymphoma", "2": "DLBCL", "3": "High-grade lymphoma", "4": "Other lymphoma subtype"},
    "prior_treatment_lines_alt": {"1": "1-2 prior therapy lines", "2": ">=3 prior therapy lines"},
    "hypogammaglobulinemia": {"0": "No hypogammaglobulinemia", "1": "Hypogammaglobulinemia"},
    "corticosteroid_use": {"0": "No corticosteroid exposure", "1": "Corticosteroid exposure"},
    "tocilizumab_use": {"0": "No tocilizumab exposure", "1": "Tocilizumab exposure"},
    "ivig_use": {"0": "No IVIG replacement", "1": "IVIG replacement"},
    "crs_before_infection": {"0": "No CRS before infection/day 100", "1": "CRS before infection/day 100"},
    "icans_before_infection": {"0": "No ICANS before infection/day 100", "1": "ICANS before infection/day 100"},
    "crs_grade": {"0": "No CRS", "1": "CRS grade 1", "2": "CRS grade 2", "3": "CRS grade 3", "4": "CRS grade 4", "5": "CRS grade 5"},
    "icans_grade": {
        "0": "No ICANS/neurotoxicity",
        "1": "ICANS grade 1",
        "2": "ICANS grade 2",
        "3": "ICANS grade 3",
        "4": "ICANS grade 4",
        "5": "ICANS grade 5",
    },
    "crs_grade_group": {"0": "No CRS", "1": "CRS grade 1-2", "2": "CRS grade >=3"},
    "icans_grade_group": {"0": "No ICANS", "1": "ICANS grade 1-2", "2": "ICANS grade >=3"},
}


FEATURE_LABEL_OVERRIDES = {
    "lymphoma_disease_list": {
        "short_label": "Conditioning regimen",
        "full_label": "Lymphodepletion/conditioning regimen list",
    }
}


REGIMEN_TERMS = {
    "CT_ARAC": "cytarabine",
    "CT_BEND": "bendamustine",
    "CT_CARB": "carboplatin",
    "CT_CY": "cyclophosphamide",
    "CT_ETOP": "etoposide",
    "CT_FLUD": "fludarabine",
    "CT_OTHDRG": "other drug",
}


@st.cache_resource
def load_bundle():
    install_sklearn_pickle_compat()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_feature_labels() -> pd.DataFrame:
    if not TABLE_PATH.exists():
        return pd.DataFrame(columns=["feature", "short_label", "full_label", "stage"])
    labels = pd.read_excel(TABLE_PATH, sheet_name="feature_label_map")
    labels = labels[["feature", "short_label", "full_label", "stage"]].drop_duplicates("feature")
    return labels


@st.cache_data
def load_reference_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TABLE_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()
    table3 = pd.read_excel(TABLE_PATH, sheet_name="table3_model_voting_performance")
    table4 = pd.read_excel(TABLE_PATH, sheet_name="table4_95pct_feature_selection")
    return table3, table4


def get_model_features(bundle: dict) -> list[str]:
    members = bundle["selected_ensemble"]["members"].split("; ")
    needed = []
    for member in members:
        needed.extend(bundle["base_models"][member]["features"])
    return sorted(set(needed))


def feature_catalog(bundle: dict) -> pd.DataFrame:
    labels = load_feature_labels()
    label_map = labels.set_index("feature").to_dict("index") if not labels.empty else {}
    metadata = load_model_metadata()
    selected_features = set(bundle.get("selected_features", []) or metadata.get("selected_features", []))
    if not selected_features and "XGB_combined_selected_xgb" in bundle.get("base_models", {}):
        selected_features = set(bundle["base_models"]["XGB_combined_selected_xgb"]["features"])
    defaults = bundle["all_feature_defaults"]
    rows = []
    for feature in get_model_features(bundle):
        meta = defaults[feature]
        label_info = label_map.get(feature, {})
        label_info.update(FEATURE_LABEL_OVERRIDES.get(feature, {}))
        rows.append(
            {
                "feature": feature,
                "short_label": label_info.get("short_label", feature),
                "full_label": label_info.get("full_label", feature),
                "stage": label_info.get("stage", "unknown"),
                "type": meta["type"],
                "required_for_primary_feature_set": feature in selected_features,
            }
        )
    catalog = pd.DataFrame(rows)
    catalog["stage_rank"] = catalog["stage"].map(STAGE_ORDER).fillna(3)
    return catalog.sort_values(
        ["required_for_primary_feature_set", "stage_rank", "short_label"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def is_binary_categories(categories: list[str]) -> bool:
    values = {str(x) for x in categories}
    return values.issubset({"0.0", "1.0", "0", "1", "__MISSING__"})


def canonical_code(value) -> str:
    value = str(value)
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return value


def format_code(value: str) -> str:
    code = canonical_code(value)
    return "missing" if code == "__MISSING__" else code


def regimen_label(value: str) -> str:
    if value == "Not specified":
        return "Conditioning regimen not specified"
    parts = [part.strip() for part in value.split("+")]
    labels = [REGIMEN_TERMS.get(part, part) for part in parts]
    return " + ".join(labels)


def display_value(feature: str, value: str, categories: list[str]) -> str:
    value = str(value)
    if value == "__MISSING__":
        return "Missing / not available"
    code = canonical_code(value)
    if feature in VALUE_LABELS and code in VALUE_LABELS[feature]:
        return f"{VALUE_LABELS[feature][code]} [code {format_code(value)}]"
    if feature == "lymphoma_disease_list":
        return f"{regimen_label(value)} [{value}]"
    if feature in {"time_to_crs_day", "time_to_icans_day"} and value not in {"__MISSING__"}:
        return f"Day {value.replace('.0', '')}"
    if feature == "cart_cell_dose":
        try:
            return f"{float(value):,.0f} cells"
        except ValueError:
            return value
    if len(categories) > 25:
        return value
    return f"Registry code {format_code(value)}"


def option_maps(feature: str, categories: list[str]) -> tuple[list[str], dict[str, str]]:
    labels = []
    reverse = {}
    used = set()
    for value in categories:
        label = display_value(feature, str(value), categories)
        if label in used:
            label = f"{label} [{value}]"
        used.add(label)
        labels.append(label)
        reverse[label] = str(value)
    return labels, reverse


def predict_risk(bundle: dict, row: pd.DataFrame) -> float:
    selected = bundle["selected_ensemble"]
    members = selected["members"].split("; ")
    member_scores = []
    for member in members:
        info = bundle["base_models"][member]
        features = info["features"]
        score = float(info["model"].predict_proba(row[features])[:, 1][0])
        member_scores.append(score)

    if selected["ensemble"].startswith("soft_weighted"):
        weights = np.array([bundle["base_model_oof_auprc"][member] for member in members], dtype=float)
        weights = weights / weights.sum()
        risk = float(np.average(member_scores, weights=weights))
    else:
        risk = float(np.mean(member_scores))
    return risk


def predict_row(bundle: dict, row: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    selected = bundle["selected_ensemble"]
    members = selected["members"].split("; ")
    detail_rows = []
    member_scores = []
    for member in members:
        info = bundle["base_models"][member]
        features = info["features"]
        score = float(info["model"].predict_proba(row[features])[:, 1][0])
        member_scores.append(score)
        detail_rows.append({"Model": member, "Risk probability": score})

    risk = predict_risk(bundle, row)
    return risk, pd.DataFrame(detail_rows)


def make_default_row(bundle: dict) -> dict:
    return {feature: meta["default"] for feature, meta in bundle["all_feature_defaults"].items()}


def normalize_uploaded_frame(bundle: dict, uploaded: pd.DataFrame) -> pd.DataFrame:
    catalog = feature_catalog(bundle)
    label_to_feature = {
        row["short_label"]: row["feature"]
        for _, row in catalog.iterrows()
    }
    label_to_feature.update(
        {
            row["full_label"]: row["feature"]
            for _, row in catalog.iterrows()
        }
    )
    renamed = uploaded.rename(columns={col: label_to_feature.get(col, col) for col in uploaded.columns})
    defaults = make_default_row(bundle)
    out = pd.DataFrame([defaults.copy() for _ in range(len(renamed))])
    for feature in defaults:
        if feature in renamed.columns:
            out[feature] = renamed[feature].fillna(defaults[feature]).astype(str)
    for feature, meta in bundle["all_feature_defaults"].items():
        if meta["type"] == "numeric":
            out[feature] = pd.to_numeric(out[feature], errors="coerce").fillna(float(meta["default"]))
    return out


def dataframe_to_xlsx(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
    return bio.getvalue()


def format_feature_value(feature: str, value, meta: dict) -> str:
    if meta["type"] == "numeric":
        return f"{float(value):.3g}"
    return display_value(feature, str(value), [str(x) for x in meta["categories"]])


def local_voting_shap(
    bundle: dict,
    row: pd.DataFrame,
    catalog: pd.DataFrame,
    n_permutations: int = 64,
) -> tuple[pd.DataFrame, float, float, float]:
    features = get_model_features(bundle)
    baseline = pd.DataFrame([make_default_row(bundle)])
    for feature, meta in bundle["all_feature_defaults"].items():
        if meta["type"] == "numeric":
            baseline[feature] = pd.to_numeric(baseline[feature], errors="coerce")

    base_risk = predict_risk(bundle, baseline)
    final_risk = predict_risk(bundle, row)
    phi = {feature: 0.0 for feature in features}
    rng = np.random.default_rng(20260603)

    for _ in range(n_permutations):
        permuted = list(rng.permutation(features))
        working = baseline.copy()
        previous = base_risk
        for feature in permuted:
            working.loc[:, feature] = row[feature].iloc[0]
            current = predict_risk(bundle, working)
            phi[feature] += current - previous
            previous = current

    label_map = catalog.set_index("feature").to_dict("index")
    rows = []
    for feature in features:
        meta = bundle["all_feature_defaults"][feature]
        info = label_map.get(feature, {})
        contribution = phi[feature] / n_permutations
        patient_value = row[feature].iloc[0]
        reference_value = baseline[feature].iloc[0]
        rows.append(
            {
                "Feature": info.get("short_label", feature),
                "Feature group": "Primary selected" if info.get("required_for_primary_feature_set", False) else "Additional reproduction",
                "Patient value": format_feature_value(feature, patient_value, meta),
                "Reference value": format_feature_value(feature, reference_value, meta),
                "Contribution": contribution,
                "Direction": "Increases risk" if contribution > 0 else ("Decreases risk" if contribution < 0 else "Neutral"),
                "Different from reference": str(patient_value) != str(reference_value),
            }
        )
    shap_df = pd.DataFrame(rows).sort_values("Contribution", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    residual = final_risk - base_risk - float(shap_df["Contribution"].sum())
    return shap_df, base_risk, final_risk, residual


def plot_local_shap(shap_df: pd.DataFrame, max_display: int = 20):
    nonzero = shap_df[shap_df["Contribution"].abs() > 1e-8].copy()
    plot_df = nonzero.head(max_display).iloc[::-1].copy()
    colors = np.where(plot_df["Contribution"] >= 0, "#b84a4a", "#3b6fb6")
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.38 * max(len(plot_df), 1))), dpi=160)
    if plot_df.empty:
        ax.text(
            0.5,
            0.5,
            "No non-zero local contribution:\npatient inputs match the reference profile.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#475569",
        )
        ax.set_yticks([])
    else:
        bars = ax.barh(plot_df["Feature"], plot_df["Contribution"], color=colors, alpha=0.86)
        xmax = float(plot_df["Contribution"].abs().max())
        pad = max(xmax * 0.08, 0.002)
        for bar, value in zip(bars, plot_df["Contribution"]):
            xpos = value + (pad if value >= 0 else -pad)
            ax.text(
                xpos,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.3f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9,
                color="#111827",
            )
        ax.set_xlim(-xmax * 1.25 - pad, xmax * 1.25 + pad)
    ax.axvline(0, color="#475569", linewidth=0.9)
    ax.set_xlabel("Contribution to final voting-model risk")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="CAR-T Infection Risk Prescreening", layout="wide")

st.title("CAR-T Day-100 Infection Risk Prescreening")
st.caption(
    "Training-cohort-derived research-use prescreening calculator for day-100 infection risk stratification after CD19 CAR-T therapy. "
    "Outputs require local recalibration and prospective workflow evaluation before any clinical implementation."
)

if not MODEL_PATH.exists():
    st.error("Model bundle not found. Run the modelling pipeline first.")
    st.stop()

bundle = load_bundle()
catalog = feature_catalog(bundle)
table3, table4 = load_reference_tables()
selected = bundle["selected_ensemble"]
threshold = float(selected["threshold"])

with st.sidebar:
    st.header("Model")
    st.metric("Threshold", f"{threshold:.2f}")
    st.metric("Internal AUROC", f"{selected['test_auroc']:.3f}")
    st.metric("Internal AUPRC", f"{selected['test_auprc']:.3f}")
    st.metric("Internal NPV", f"{selected['test_npv']:.3f}")
    st.write("Primary method: AUPRC-weighted soft voting.")
    st.write("Held-out training-cohort attention/not-low-risk group rate: " f"{selected['test_flagged_rate']:.3f}.")
    st.caption("The score is intended for research-use stratification, not calibrated individual treatment decisions.")

input_tab, batch_tab, model_tab = st.tabs(["Single patient", "Batch xlsx", "Model documentation"])

with input_tab:
    st.subheader("Single-patient calculator")
    st.write(
        "Primary selected variables are shown first. Additional variables are retained only for full reproduction "
        "of the fixed voting ensemble."
    )

    values = make_default_row(bundle)
    primary = catalog[catalog["required_for_primary_feature_set"]]
    advanced = catalog[~catalog["required_for_primary_feature_set"]]

    with st.form("single_patient_form"):
        for title, subset, expanded in [
            ("Primary selected variables", primary, True),
            ("Additional variables for full voting-model reproduction", advanced, False),
        ]:
            with st.expander(title, expanded=expanded):
                cols = st.columns(3)
                for idx, row in subset.iterrows():
                    feature = row["feature"]
                    meta = bundle["all_feature_defaults"][feature]
                    label = row["short_label"]
                    help_text = f"{row['full_label']} | stage: {row['stage']} | raw field: {feature}"
                    with cols[idx % 3]:
                        if meta["type"] == "numeric":
                            values[feature] = st.number_input(
                                label,
                                value=float(meta["default"]),
                                min_value=float(meta["min"]),
                                max_value=float(meta["max"]),
                                help=help_text,
                            )
                        else:
                            categories = [str(x) for x in meta["categories"]]
                            option_labels, reverse = option_maps(feature, categories)
                            default = str(meta["default"])
                            default_label = display_value(feature, default, categories)
                            if default_label not in reverse:
                                default_label = option_labels[0]
                            values[feature] = reverse[
                                st.selectbox(
                                    label,
                                    option_labels,
                                    index=option_labels.index(default_label),
                                    help=help_text,
                                )
                            ]
        submitted = st.form_submit_button("Calculate infection risk", type="primary")

    if submitted:
        input_row = pd.DataFrame([values])
        risk, member_df = predict_row(bundle, input_row)
        high_risk = risk >= threshold
        status_text = "Yes" if high_risk else "No"
        status_color = "#b42318" if high_risk else "#027a48"
        status_bg = "#fff1f3" if high_risk else "#ecfdf3"
        status_border = "#fda29b" if high_risk else "#75e0a7"
        status_detail = (
            "This patient is in the attention / not-low-risk group."
            if high_risk
            else "This patient is not in the attention group and may be considered a low-risk candidate."
        )
        st.markdown(
            f"""
            <div style="
                border: 1px solid {status_border};
                background: {status_bg};
                border-radius: 8px;
                padding: 18px 20px;
                margin: 8px 0 18px 0;
            ">
                <div style="font-size: 0.95rem; color: #344054; font-weight: 600;">
                    Is the patient in the attention / not-low-risk group?
                </div>
                <div style="font-size: 2.4rem; line-height: 1.15; color: {status_color}; font-weight: 800; margin-top: 4px;">
                    {status_text}
                </div>
                <div style="font-size: 0.95rem; color: #344054; margin-top: 6px;">
                    {status_detail}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        top_cols = st.columns(3)
        top_cols[0].metric("Model risk score", f"{risk:.3f}")
        top_cols[1].metric("Decision threshold", f"{threshold:.3f}")
        top_cols[2].metric("Distance from threshold", f"{risk - threshold:+.3f}")
        st.progress(min(max(risk, 0.0), 1.0))
        if high_risk:
            st.warning("Interpretation: the model score meets or exceeds the fixed training-cohort threshold; this patient is in the attention / not-low-risk group.")
        else:
            st.success("Interpretation: the model score is below the fixed training-cohort threshold; this patient is not in the attention group and may be considered a low-risk candidate.")
        st.caption(
            "Member model probabilities are shown for transparency only. "
            "The clinical grouping above is based on the final AUPRC-weighted soft-voting score and should not be interpreted as a calibrated absolute individual risk."
        )
        st.dataframe(member_df.style.format({"Risk probability": "{:.3f}"}), width="stretch")
        st.subheader("Local SHAP contribution analysis")
        st.caption(
            "This module approximates local contributions to the final AUPRC-weighted voting-model score. "
            "Contributions are relative to a training-derived reference profile; positive values increase the displayed score "
            "and negative values decrease it. These local contributions are descriptive and should not be read as causal effects."
        )
        with st.spinner("Calculating local SHAP contributions for the final voting model..."):
            shap_df, base_risk, final_risk_check, residual = local_voting_shap(bundle, input_row, catalog)
        shap_cols = st.columns(3)
        shap_cols[0].metric("Reference risk", f"{base_risk:.3f}")
        shap_cols[1].metric("Explained final risk", f"{final_risk_check:.3f}")
        shap_cols[2].metric("Approximation residual", f"{residual:+.4f}")
        has_contributions = float(shap_df["Contribution"].abs().sum()) > 1e-8
        if not has_contributions:
            st.info(
                "The current input matches the training-derived reference profile, so every local contribution is zero. "
                "Change one or more patient variables and recalculate to display the contribution plot."
            )
        else:
            st.caption(
                "The plot displays non-zero local contributions only. Red bars increase the voting-model "
                "risk; blue bars decrease it."
            )
            st.pyplot(plot_local_shap(shap_df), clear_figure=True)
        st.dataframe(
            shap_df.style.format({"Contribution": "{:+.4f}"}),
            width="stretch",
        )
        st.info("Use this output for research-use risk stratification only; do not use it as the sole basis for treatment decisions.")

with batch_tab:
    st.subheader("Batch prediction from .xlsx")
    st.write("Upload an xlsx file using raw feature names, short labels, or full labels. Missing model fields are filled with training-set defaults.")
    template = pd.DataFrame([make_default_row(bundle)])[get_model_features(bundle)]
    st.download_button(
        "Download xlsx input template",
        dataframe_to_xlsx(template),
        file_name="cart_infection_risk_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    upload = st.file_uploader("Upload xlsx file", type=["xlsx"])
    if upload is not None:
        uploaded_df = pd.read_excel(upload)
        model_df = normalize_uploaded_frame(bundle, uploaded_df)
        rows = []
        for idx in range(len(model_df)):
            risk, _ = predict_row(bundle, model_df.iloc[[idx]])
            rows.append(
                {
                    "row_id": idx + 1,
                    "voting_model_risk_score": risk,
                    "threshold": threshold,
                    "needs_attention_not_low_risk_group": "Yes" if risk >= threshold else "No",
                    "clinical_interpretation": (
                        "In attention / not-low-risk group"
                        if risk >= threshold
                        else "Not in attention group; low-risk candidate"
                    ),
                }
            )
        pred_df = pd.DataFrame(rows)
        st.dataframe(pred_df.style.format({"voting_model_risk_score": "{:.3f}", "threshold": "{:.3f}"}), width="stretch")
        st.download_button(
            "Download prediction xlsx",
            dataframe_to_xlsx(pred_df),
            file_name="cart_infection_risk_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

with model_tab:
    st.subheader("Fixed model summary")
    st.write("The model, threshold and voting rule were selected inside the training cohort and are fixed in this calculator.")
    st.dataframe(catalog.drop(columns=["stage_rank"]), width="stretch")
    if not table4.empty:
        st.subheader("Top selected features by cumulative importance")
        st.dataframe(table4.head(15), width="stretch")
    if not table3.empty:
        st.subheader("Performance of candidate voting strategies")
        display_cols = [
            "Model",
            "Threshold",
            "AUROC (95% CI)",
            "AUPRC (95% CI)",
            "Sensitivity (95% CI)",
            "NPV (95% CI)",
            "Flagged rate (95% CI)",
            "Missed infections",
            "Selected primary",
        ]
        display_table3 = table3[display_cols].rename(
            columns={"Flagged rate (95% CI)": "Attention/not-low-risk group rate (95% CI)"}
        )
        st.dataframe(display_table3, width="stretch")
