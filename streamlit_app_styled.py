import io
import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

# Make sure project root is on sys.path so we can import check.py
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

try:
    import check
    from check import Dataloader, DataAnalyzer, Preprocessor, Visualizer, DataPreparer, ModelTrainer, ModelEvaluator, ensure_discrete_target
except Exception:
    # Best-effort fallback: define minimal placeholders to avoid hard import errors during static checks.
    Dataloader = None
    DataAnalyzer = None
    Preprocessor = None
    Visualizer = None
    DataPreparer = None
    ModelTrainer = None
    ModelEvaluator = None
    def ensure_discrete_target(df, col):
        return df


def load_csv_bytes(uploaded) -> pd.DataFrame:
    """Read an uploaded file (Streamlit UploadedFile) into a DataFrame."""
    try:
        return pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        return pd.read_csv(io.TextIOWrapper(uploaded, encoding='utf-8'))


def main():
    st.set_page_config(page_title="EDA & Model Training", layout="wide")

    # Custom CSS for colors and styling
    st.markdown("""
        <style>
        .main-title { color: #1E88E5; font-size: 42px; font-weight: 600; margin-bottom: 12px; }
        .sub-title { color: #424242; font-size: 20px; margin-bottom: 24px; }
        .section-header { 
            color: #2E7D32; 
            font-size: 24px; 
            font-weight: 500;
            padding: 8px 0;
            border-bottom: 2px solid #E8F5E9;
            margin: 24px 0 16px 0;
        }
        .analysis-card {
            background: #FAFAFA;
            padding: 16px;
            border-radius: 8px;
            border-left: 4px solid #1E88E5;
            margin: 12px 0;
        }
        .success-box {
            background: #E8F5E9;
            color: #2E7D32;
            padding: 12px;
            border-radius: 6px;
            margin: 8px 0;
        }
        .info-box {
            background: #E3F2FD;
            color: #1565C0;
            padding: 12px;
            border-radius: 6px;
            margin: 8px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">📊 EDA and Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a CSV to begin exploratory analysis, preprocessing, visualization, and model comparison.</div>', unsafe_allow_html=True)

    # Sidebar: data upload
    st.sidebar.markdown("### 📁 Data Source")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"]) 
    use_sample = st.sidebar.checkbox("Use sample transfusion1.csv", value=False)

    df = None
    if uploaded_file is not None:
        try:
            df = load_csv_bytes(uploaded_file)
            st.sidebar.markdown('<div class="success-box">✅ File loaded successfully</div>', unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.error(f"Could not read uploaded CSV: {e}")
    elif use_sample:
        sample_path = os.path.join(ROOT, 'transfusion1.csv')
        if os.path.exists(sample_path):
            try:
                df = pd.read_csv(sample_path)
            except Exception as e:
                st.sidebar.error(f"Could not load sample file: {e}")
        else:
            st.sidebar.error("Sample file not found in project root")

    if df is None:
        st.markdown('<div class="info-box">ℹ️ Please upload a CSV to start or select the sample dataset.</div>', unsafe_allow_html=True)
        return

    # Save copies in session state
    st.session_state.setdefault('df_orig', df.copy())
    st.session_state.setdefault('df_proc', df.copy())

    # Sidebar: analysis options
    st.sidebar.markdown("### 🔍 Analyze")
    analyze_option = st.sidebar.selectbox("Choose an analysis view", [
        'head', 'tail', 'info', 'describe', 'columns', 'shape', 'dtypes', 'nulls', 'duplicates'
    ])

    # Sidebar: preprocessor controls (below analyze)
    st.sidebar.markdown("### ⚙️ Preprocessor")
    target = st.sidebar.selectbox("Target column (for modeling)", [None] + df.columns.tolist())
    missing_strategy = st.sidebar.selectbox("Missing values strategy", ['drop', 'fill', 'ffill', 'bfill', 'mean', 'median', 'mode'])
    fill_value = st.sidebar.text_input("Fill value (if strategy='fill')", value="")
    remove_duplicates = st.sidebar.checkbox("Remove duplicates", value=False)
    remove_columns = st.sidebar.multiselect("Drop columns (optional)", options=df.columns.tolist())
    encode_target = st.sidebar.checkbox("Encode target (if categorical)", value=True)
    scale_numeric = st.sidebar.checkbox("Scale numeric features", value=False)
    run_pre = st.sidebar.button("Apply preprocessing")

    # Main: show analysis
    st.markdown('<div class="section-header">📈 Analysis Results</div>', unsafe_allow_html=True)
    analyzer = DataAnalyzer(st.session_state['df_proc']) if DataAnalyzer else None
    if analyzer is not None:
        if analyze_option == 'head':
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.dataframe(analyzer.show('head', n=10))
            st.markdown('</div>', unsafe_allow_html=True)
        elif analyze_option == 'tail':
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.dataframe(analyzer.show('tail', n=10))
            st.markdown('</div>', unsafe_allow_html=True)
        elif analyze_option == 'info':
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            with st.capture_output():
                analyzer.show('info')
            st.write('Displayed above (info prints to stdout)')
            st.markdown('</div>', unsafe_allow_html=True)
        elif analyze_option == 'describe':
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.dataframe(analyzer.show('describe'))
            st.markdown('</div>', unsafe_allow_html=True)
        elif analyze_option == 'columns':
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.write(analyzer.show('columns'))
            st.markdown('</div>', unsafe_allow_html=True)
        elif analyze_option == 'shape':
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.write(analyzer.show('shape'))
            st.markdown('</div>', unsafe_allow_html=True)
        elif analyze_option == 'dtypes':
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.dataframe(analyzer.show('dtypes'))
            st.markdown('</div>', unsafe_allow_html=True)
        elif analyze_option == 'nulls':
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.dataframe(analyzer.show('nulls'))
            st.markdown('</div>', unsafe_allow_html=True)
        elif analyze_option == 'duplicates':
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.write(analyzer.show('duplicates'))
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.write(st.session_state['df_proc'].head(5))
        st.markdown('</div>', unsafe_allow_html=True)

    # Preprocessing action
    if run_pre:
        dfp = st.session_state['df_proc'].copy()
        if Preprocessor is None:
            st.error("Preprocessor unavailable (check.py not importable).")
        else:
            pre = Preprocessor(dfp)
            try:
                if remove_duplicates:
                    dfp = pre.remove_duplicates(inplace=False)
                if remove_columns:
                    dfp = pre.remove_columns(remove_columns, inplace=False)
                if missing_strategy:
                    fv = None
                    if missing_strategy == 'fill' and fill_value != "":
                        try:
                            fv = float(fill_value)
                        except Exception:
                            fv = fill_value
                    dfp = pre.handle_missing_values(strategy=missing_strategy, fill_value=fv, columns=None, inplace=False)
                if encode_target and target in dfp.columns:
                    dfp = pre.encode_categorical(target_column=target, inplace=False)
                if scale_numeric:
                    num_cols = dfp.select_dtypes(include=['number']).columns.tolist()
                    if target in num_cols:
                        num_cols.remove(target)
                    if num_cols:
                        dfp = pre.scale_numeric(columns=num_cols, inplace=False)

                st.session_state['df_proc'] = dfp
                st.markdown('<div class="success-box">✨ Preprocessing applied successfully</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")

    # Show cleaned dataframe
    st.markdown('<div class="section-header">🔄 Cleaned Dataset Preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.dataframe(st.session_state['df_proc'].head(200))
    st.markdown('</div>', unsafe_allow_html=True)

    # Plots controls
    st.markdown('<div class="section-header">📊 Visualization</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📈 Univariate Analysis")
        uni_col = st.selectbox("Univariate column", st.session_state['df_proc'].columns.tolist(), key='u_col')
        uni_plot = st.selectbox("Univariate plot", ['histogram', 'boxplot', 'bar'], key='u_type')
        if st.button("Show univariate plot"):
            try:
                if uni_plot == 'histogram':
                    fig = px.histogram(st.session_state['df_proc'], x=uni_col, nbins=40)
                elif uni_plot == 'boxplot':
                    fig = px.box(st.session_state['df_proc'], y=uni_col)
                else:
                    vc = st.session_state['df_proc'][uni_col].value_counts().reset_index()
                    vc.columns = [uni_col, 'count']
                    fig = px.bar(vc.head(100), x=uni_col, y='count')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Plotting failed: {e}")
    with col2:
        st.markdown("#### 📉 Bivariate Analysis")
        bi_x = st.selectbox("Bivariate X", st.session_state['df_proc'].columns.tolist(), key='b_x')
        bi_y = st.selectbox("Bivariate Y", st.session_state['df_proc'].columns.tolist(), key='b_y')
        bi_plot = st.selectbox("Bivariate plot", ['scatter','hex','reg','box','violin'], key='b_type')
        if st.button("Show bivariate plot"):
            try:
                dfp = st.session_state['df_proc']
                if bi_plot == 'scatter':
                    fig = px.scatter(dfp, x=bi_x, y=bi_y, color=target if target in dfp.columns else None)
                elif bi_plot == 'hex':
                    fig = px.density_heatmap(dfp, x=bi_x, y=bi_y)
                elif bi_plot == 'reg':
                    fig = px.scatter(dfp, x=bi_x, y=bi_y, trendline='ols')
                elif bi_plot == 'box':
                    fig = px.box(dfp, x=bi_x, y=bi_y)
                else:
                    fig = px.violin(dfp, x=bi_x, y=bi_y)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Plotting failed: {e}")

    # Modeling
    st.markdown('<div class="section-header">🤖 Modeling & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Train simple classifiers and compare their performance</div>', unsafe_allow_html=True)
    test_frac = st.slider("Test %", 10, 50, 30) / 100.0
    run_models = st.button("Train & Evaluate")

    if run_models:
        if target is None:
            st.error("Please select a target column in the sidebar before training")
        elif target not in st.session_state['df_proc'].columns:
            st.error("Selected target not present after preprocessing")
        else:
            dfp = st.session_state['df_proc']
            X = dfp.drop(columns=[target])
            y = dfp[target]
            # Ensure target is discrete (helper)
            try:
                dfp = ensure_discrete_target(dfp, target)
                y = dfp[target]
            except Exception:
                pass

            if DataPreparer is None:
                st.error("Modeling utilities unavailable (check.py not importable).")
            else:
                dp = DataPreparer(X, y, test_size=test_frac)
                X_train, X_test, y_train, y_test = dp.split_data()
                trainer = ModelTrainer(X_train, y_train)
                with st.spinner('🔄 Training models...'):
                    models = trainer.train_models()
                evaluator = ModelEvaluator(models, X_test, y_test)
                with st.spinner('📊 Evaluating models...'):
                    results = evaluator.evaluate_models()
                best_name, best_acc = evaluator.find_best_model()

                # Show accuracies and suggest best
                st.markdown('### 📊 Model Performance', unsafe_allow_html=True)
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                for name, res in results.items():
                    st.write(f"{name}: {res['accuracy']:.4f}")
                if best_name:
                    st.markdown(f'<div class="success-box">🏆 Best model: {best_name} (accuracy={best_acc:.4f})</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()