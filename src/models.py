import pandas as pd
import numpy as np
from scipy import stats

import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

# --- 1. HELPER FUNCTION ---

def _get_days_to_nearest_event(dates, event_dates):
    event_dates_ts = pd.to_datetime(event_dates, utc=True).sort_values().to_series().drop_duplicates()
    dates = pd.to_datetime(dates, utc=True)
    
    next_event_indices = event_dates_ts.searchsorted(dates, side='left')
    next_event_dates_series = event_dates_ts.iloc[np.clip(next_event_indices, 0, len(event_dates_ts) - 1)]
    
    dt_next = np.array(pd.to_timedelta(next_event_dates_series.values - dates.values).days, dtype=float)
    mask_next = (dates > event_dates_ts.max()).values
    dt_next[mask_next] = np.inf

    prev_event_indices = event_dates_ts.searchsorted(dates, side='right') - 1
    prev_event_dates_series = event_dates_ts.iloc[np.clip(prev_event_indices, 0, len(event_dates_ts) - 1)]
    
    dt_prev = np.array(pd.to_timedelta(dates.values - prev_event_dates_series.values).days, dtype=float)
    mask_prev = (dates < event_dates_ts.min()).values
    dt_prev[mask_prev] = np.inf

    signed_days = np.where(
        dt_next <= dt_prev, 
        -dt_next,
        dt_prev
    )
    return signed_days

# --- 2. WILCOXON TEST FUNCTION ---

def analyze_with_wilcoxon_test(df, referendum_dates, period_days=30):
    print("\n--- Performing Paired Analysis (Wilcoxon Signed-Rank Test) ---")
    before_views_agg = []
    after_views_agg = []
    referendum_dates = pd.to_datetime(referendum_dates, utc=True)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    for ref_date in referendum_dates:
        before_period = pd.date_range(end=ref_date - pd.Timedelta(days=1), periods=period_days, tz='UTC')
        after_period = pd.date_range(start=ref_date + pd.Timedelta(days=1), periods=period_days, tz='UTC')
        before_data = df[df['date'].isin(before_period)]
        after_data = df[df['date'].isin(after_period)]
        
        if len(before_data) == period_days and len(after_data) == period_days:
            before_views_agg.extend(before_data['ad_views'].tolist())
            after_views_agg.extend(after_data['ad_views'].tolist())
            
    if len(before_views_agg) < 10: 
        print(f"--- Insufficient Paired Data for Wilcoxon Test ({len(before_views_agg)} pairs) ---")
    else:
        wilcoxon_test = stats.wilcoxon(before_views_agg, after_views_agg, alternative='greater', zero_method='zsplit')
        print(f"Wilcoxon signed-rank test (n={len(before_views_agg)} pairs): Statistic={wilcoxon_test.statistic:.3f}, p-value={wilcoxon_test.pvalue}")


# --- 3. GAM ANALYSIS FUNCTION ---

def analyze_with_gam(daily_df, referendum_dates, target_variable='ad_views', window_days=60, analysis_name='Ad Activity'):
    print(f"\n--- Performing GAM (Generalized Additive Model) Analysis for {analysis_name} (Window: +/- {window_days} days) ---")
    
    try:
        from pygam import LinearGAM, s, f
    except ImportError:
        print("FATAL ERROR: pygam not installed. pip install pygam")
        return None, None, None

    df_gam = daily_df.copy()
    df_gam[target_variable] = pd.to_numeric(df_gam[target_variable], errors='coerce')
    df_gam = df_gam.dropna(subset=[target_variable])

    print("Engineering GAM features (days_to_ref, seasonality, trend)...")
    df_gam['date'] = pd.to_datetime(df_gam['date'], utc=True)
    
    df_gam['days_to_ref'] = _get_days_to_nearest_event(df_gam['date'], referendum_dates)
    df_gam['day_of_week'] = df_gam['date'].dt.dayofweek
    df_gam['day_of_year'] = df_gam['date'].dt.dayofyear
    df_gam['time_trend'] = (df_gam['date'] - df_gam['date'].min()).dt.days
    
    df_model = df_gam[df_gam['days_to_ref'].abs() <= window_days].copy()
    
    if len(df_model) < 100: 
         print(f"Skipping GAM: Not enough data within {window_days}-day window ({len(df_model)} days).")
         return None, None, None
         
    try:
        term_0 = s(0, n_splines=20)
        term_1 = s(1, n_splines=12, basis='cp')
        term_2 = f(2)
        term_3 = s(3, n_splines=8)
        gam_terms = term_0 + term_1 + term_2 + term_3
        
        X = df_model[['days_to_ref', 'day_of_year', 'day_of_week', 'time_trend']]
        y = np.log1p(df_model[target_variable])
        
        print(f"Fitting GAM on {len(X)} data points...")
        
        gam = LinearGAM(gam_terms).fit(X, y)
        
        # print(f"\nGAM analysis for target '{analysis_name}':")
        # gam.summary() # Suppress p-value warnings
        
        print("Calculating partial dependence for 'days_to_ref'...")
        XX = gam.generate_X_grid(term=0, n=100) 
        pdep, confs = gam.partial_dependence(term=0, X=XX, width=0.95)
        
        return XX[:, 0], pdep, confs
        
    except Exception as e:
        print(f"Error during GAM analysis: {e}")
        return None, None, None


def run_full_analysis(df, referendums, analysis_window_days=30):

    print("Starting full analysis loop...")

    spending_gam_results = {}
    impressions_gam_results = {}
    
    lang_groups = {
        'italian': ['italian'],
        'german': ['german'],
        'french': ['french'],
        'all': ['italian', 'german', 'french']
    }

    for lang_name, lang_list in lang_groups.items():
        print(f"\n####################################################")
        print(f"--- Analyzing Languages: {lang_list} (as '{lang_name}') ---")
        print(f"####################################################")

        df_lang = df[df['lang'].isin(lang_list)]
        if df_lang.empty:
            print("No data for this language group. Skipping.")
            continue

        # --- 3a. Aggregate your RAW data to DAILY data ---
        temp_agg = df_lang.groupby(['date']).agg({
            'spend_avg': 'sum',
            'impressions_avg': 'sum'
        }).reset_index()
        
        if temp_agg.empty:
            print("No data for this group. Skipping.")
            continue
            
        temp_agg.set_index('date', inplace=True)
        
        # --- 3b. Create complete daily dataframe (one row per day) ---
        min_date = temp_agg.index.min()
        max_date = temp_agg.index.max()
        full_date_range = pd.date_range(start=min_date, end=max_date, tz='UTC')
        
        df_complete = temp_agg.reindex(full_date_range, fill_value=0).reset_index()
        df_complete = df_complete.rename(columns={'index': 'date'})
        
        # --- 4. Run Analysis for SPENDING ('spend_avg') ---
        print("\n\n===== ANALYSIS FOR: SPENDING =====")
        df_spend = df_complete.rename(columns={'spend_avg': 'ad_views'})
        
        analyze_with_wilcoxon_test(df_spend.copy(), referendums, period_days=30)
        x_s, y_s, conf_s = analyze_with_gam(
            df_spend.copy(), 
            referendums, 
            target_variable='ad_views', 
            window_days=analysis_window_days, 
            analysis_name=f"{lang_name.upper()} Spending"
        )
        if x_s is not None:
            spending_gam_results[lang_name] = (x_s, y_s, conf_s)

        # --- 5. Run Analysis for IMPRESSIONS ('impressions_avg') ---
        print("\n\n===== ANALYSIS FOR: IMPRESSIONS =====")
        df_impressions = df_complete.rename(columns={'impressions_avg': 'ad_views'})
        
        analyze_with_wilcoxon_test(df_impressions.copy(), referendums, period_days=30)
        x_i, y_i, conf_i = analyze_with_gam(
            df_impressions.copy(), 
            referendums, 
            target_variable='ad_views', 
            window_days=analysis_window_days, 
            analysis_name=f"{lang_name.upper()} Impressions"
        )
        if x_i is not None:
            impressions_gam_results[lang_name] = (x_i, y_i, conf_i)

        print(f"\n--- Finished analysis for {lang_list} ---\n")
        
    return spending_gam_results, impressions_gam_results, analysis_window_days

def test_stat(df):
    df = df.dropna(subset=['date'])
    
    # --- 2. Data Preparation for Testing ---
    # Separate the data into four distinct groups based on the referendum outcome
    # and the campaign side.
    yes_approved = df[df['approved'] == 'Accepted']['Impressions yes'].dropna()
    yes_not_approved = df[df['approved'] == 'Rejected']['Impressions yes'].dropna()
    
    no_approved = df[df['approved'] == 'Accepted']['Impressions no'].dropna()
    no_not_approved = df[df['approved'] == 'Rejected']['Impressions no'].dropna()
    
    
    # --- 3. Statistical Testing ---
    # The Mann-Whitney U test is a non-parametric test used to determine if two
    # independent samples were drawn from populations with the same distribution.
    
    print("--- Statistical Test Results ---")
    
    # Test 1: Correlation between high 'Impressions yes' and 'approved == Yes'
    # H0 (Null Hypothesis): The distribution of 'Impressions yes' is the same for both 'Yes' and 'No' outcomes.
    # Ha (Alternative Hypothesis): The distribution of 'Impressions yes' is greater for 'Yes' outcomes.
    stat1, p_value1 = stats.mannwhitneyu(yes_approved, yes_not_approved, alternative='greater')
    
    print(f"\nTest 1: Are 'Impressions yes' significantly higher for approved referendums?")
    print(f"Mann-Whitney U statistic: {stat1:.2f}")
    print(f"P-value: {p_value1:.4f}")
    
    if p_value1 < 0.05:
        print("Result: The difference is statistically significant. We reject the null hypothesis.")
    else:
        print("Result: The difference is not statistically significant. We fail to reject the null hypothesis.")
    
    
    # Test 2: Correlation between high 'Impressions no' and 'approved == No'
    # H0 (Null Hypothesis): The distribution of 'Impressions no' is the same for both 'Yes' and 'No' outcomes.
    # Ha (Alternative Hypothesis): The distribution of 'Impressions no' is greater for 'No' outcomes.
    stat2, p_value2 = stats.mannwhitneyu(no_not_approved, no_approved, alternative='greater')
    
    print(f"\nTest 2: Are 'Impressions no' significantly higher for non-approved referendums?")
    print(f"Mann-Whitney U statistic: {stat2:.2f}")
    print(f"P-value: {p_value2:.4f}")
    
    if p_value2 < 0.05:
        print("Result: The difference is statistically significant. We reject the null hypothesis.")
    else:
        print("Result: The difference is not statistically significant. We fail to reject the null hypothesis.")

def get_prediction_model(df_party, feature_cols, weights=True):
    """
    Builds and fits the multinomial logistic regression model.
    """
    X = df_party[feature_cols]
    X = sm.add_constant(X, prepend=False) # Added prepend=False, common practice
    y_true = df_party['party_code']

    model = sm.MNLogit(y_true, X)

    if weights:
        # print("Fitting model with weights...")
        weight_values = df_party['impressions_avg']
        fit = model.fit_regularized(freq_weights=weight_values, maxiter=5000)
    else:
        # print("Fitting model without weights...")
        fit = model.fit_regularized(freq_weights=None, maxiter=5000) 
        
    return fit, X, y_true

def get_top_significant_coeffs(fit, party_mapping, p_value_threshold=0.05):
    """
    Extracts the top 3 most impactful, statistically significant coefficients 
    for each party from a fitted statsmodels MNLogit model,
    correctly handling the baseline category.
    """
    # Identify the baseline party (the one with code 0)
    base_party_name = party_mapping[0]
    print(f"Note: '{base_party_name}' is the base category for comparison.\n")

    column_labels = [party_mapping[i+1] for i in range(fit.params.shape[1])]
    coeffs = pd.DataFrame(fit.params)
    coeffs = coeffs.rename(columns={
        i: party_mapping[i+1] for i in range(fit.params.shape[1])
    })
    p_values = pd.DataFrame(fit.pvalues)
    p_values = p_values.rename(columns={
        i: party_mapping[i+1] for i in range(fit.params.shape[1])
    })    
    results = {}
    for party in coeffs.columns:
        # Combine coeffs and p-values for the current party
        party_df = pd.DataFrame({
            'Coefficient': coeffs[party],
            'P-Value': p_values[party]
        }).drop('const') # Exclude the intercept

        # Filter for statistical significance
        significant = party_df[party_df['P-Value'] < p_value_threshold]
        
        # Get the top 3 by the absolute magnitude of the coefficient
        top_3 = significant.reindex(significant.Coefficient.abs().sort_values(ascending=False).index).head(3)
        
        results[party] = top_3
        
    return results

def get_top_significant_odds_ratios(fit, party_mapping, p_value_threshold=0.05, top_head=3):
    """
    Calculates odds ratios and extracts the top 3 most impactful and 
    statistically significant ones for each party.
    """
    results = {}

    column_labels = [party_mapping[i+1] for i in range(fit.params.shape[1])]
    coeffs = pd.DataFrame(fit.params)
    coeffs = coeffs.rename(columns={
        i: party_mapping[i+1] for i in range(fit.params.shape[1])
    })
    p_values = pd.DataFrame(fit.pvalues)
    p_values = p_values.rename(columns={
        i: party_mapping[i+1] for i in range(fit.params.shape[1])
    })

    for party in coeffs.columns:
        party_df = pd.DataFrame({
            'Coefficient': coeffs[party],
            'P-Value': p_values[party]
        }).drop('const')

        significant = party_df[party_df['P-Value'] < p_value_threshold].copy()
        
        if not significant.empty:
            # Calculate Odds Ratio
            significant['Odds Ratio'] = np.exp(significant['Coefficient'])
            
            # Sort by odds ratio to find the strongest positive predictors
            top_3 = significant.sort_values(by='Odds Ratio', ascending=False).head(top_head)
            results[party] = top_3
        
    return results

def get_top_bottom_mnlogit_margeff(
    fit,
    party_mapping: dict,
    p_value_threshold: float = 0.05,
    top_n: int = 3
) -> pd.DataFrame:
    """
    Extracts the top N most positive and top N most negative *marginal effects*
    for each party from a fitted MNLogit model.

    Marginal effects represent the change in the probability of an outcome
    for a one-unit change in a predictor variable.

    Parameters
    ----------
    fit : statsmodels.discrete.discrete_model.MNLogitResults
        A fitted MNLogit model object.
    party_mapping : dict
        A mapping from the integer codes of the dependent variable to
        party/class names (e.g., {0: 'Party A', 1: 'Party B'}).
    p_value_threshold : float, default 0.05
        The significance level for filtering the marginal effects.
    top_n : int, default 3
        The number of top positive and top negative effects to return for each party.

    Returns
    -------
    results_df : pd.DataFrame
        A tidy DataFrame with the following columns:
        ['Party', 'Feature', 'Marginal Effect', 'P-Value', 'Effect'],
        where Effect is either 'Top Positive' or 'Top Negative'.
    """
    # 1. Calculate marginal effects and get the summary DataFrame
    margeff = fit.get_margeff()
    margeff_df = margeff.summary_frame()
    margeff_df = margeff_df.reset_index()
    # 2. Rename columns based on the actual output
    # The key columns are 'endog', 'exog', 'dy/dx', and 'Pr(>|z|)'
    margeff_df = margeff_df.rename(columns={
        'endog': 'Party',
        'exog': 'Feature',
        'dy/dx': 'Marginal Effect',
        'Pr(>|z|)': 'P-Value'
    })

    # 3. Parse the 'Party' column to extract the numeric code
    # It extracts the number from strings like 'party_code=0'
    party_codes = margeff_df['Party'].str.extract(r'(\d+)')[0].astype(int)
    
    # Map these numeric codes to the actual party names
    margeff_df['Party'] = party_codes.map(party_mapping)

    # Filter out the constant/intercept if it exists
    margeff_df = margeff_df[margeff_df['Feature'] != 'const']

    results_list = []

    # 4. Loop through each party to find top/bottom effects
    # Ensure all parties from the mapping are considered
    for party_name in party_mapping.values():
        party_df = margeff_df[margeff_df['Party'] == party_name]
        
        if party_df.empty:
            continue

        # Filter by p-value significance
        significant = party_df[party_df['P-Value'] <= p_value_threshold].copy()
        if significant.empty:
            continue

        # 5. Get top N positive and negative effects
        top_pos = significant.sort_values(by='Marginal Effect', ascending=False).head(top_n)
        top_pos['Effect'] = 'Top Positive'

        top_neg = significant.sort_values(by='Marginal Effect', ascending=True).head(top_n)
        top_neg['Effect'] = 'Top Negative'

        results_list.extend([top_pos, top_neg])

    if not results_list:
        print("Warning: No significant marginal effects found at the given p-value threshold.")
        return pd.DataFrame()

    # 6. Concatenate results and select final columns
    results_df = pd.concat(results_list, ignore_index=True)
    
    # Ensure the correct columns are selected in the final output
    final_cols = ['Party', 'Feature', 'Marginal Effect', 'P-Value', 'Effect']
    results_df = results_df[final_cols]

    return results_df

def get_top_bottom_mnlogit_effects(
    fit, 
    party_mapping, 
    p_value_threshold=0.05, 
    top_n=3
):
    """
    Extracts top N positive and top N negative coefficients for each party
    from a fitted MNLogit model, including odds ratios.

    Parameters
    ----------
    fit : statsmodels.discrete.discrete_model.MNLogitResults
        Fitted MNLogit model.
    party_mapping : dict
        Mapping from integer codes to party/class names.
    p_value_threshold : float, default 0.05
        Significance level for filtering coefficients.
    top_n : int, default 3
        Number of top and bottom coefficients to return for each party.

    Returns
    -------
    results_df : pd.DataFrame
        Tidy table with columns:
        ['Party', 'Feature', 'Coefficient', 'Odds Ratio', 'P-Value', 'Effect']
        where Effect ∈ {'Top Positive', 'Top Negative'}
    """
    base_party_name = party_mapping[0]
    print(f"Note: '{base_party_name}' is the base category for comparison.\n")

    # Prepare coefficient and p-value DataFrames
    coeffs = pd.DataFrame(fit.params)
    coeffs = coeffs.rename(columns={i: party_mapping[i+1] for i in range(fit.params.shape[1])})
    p_values = pd.DataFrame(fit.pvalues)
    p_values = p_values.rename(columns={i: party_mapping[i+1] for i in range(fit.params.shape[1])})

    results_list = []

    for party in coeffs.columns:
        party_df = pd.DataFrame({
            'Feature': coeffs.index,
            'Coefficient': coeffs[party],
            'P-Value': p_values[party]
        })
        party_df = party_df[party_df['Feature'] != 'const']

        # Filter by p-value significance
        significant = party_df[party_df['P-Value'] <= p_value_threshold].copy()
        if significant.empty:
            continue

        # Compute odds ratios
        significant['Odds Ratio'] = np.exp(significant['Coefficient'])

        # Sort by coefficient
        top_pos = significant.sort_values(by='Coefficient', ascending=False).head(top_n)
        top_pos['Effect'] = 'Top Positive'

        top_neg = significant.sort_values(by='Coefficient', ascending=True).head(top_n)
        top_neg['Effect'] = 'Top Negative'

        # Tag party
        top_pos['Party'] = party
        top_neg['Party'] = party

        results_list.extend([top_pos, top_neg])

    results_df = pd.concat(results_list, ignore_index=True)
    results_df = results_df[['Party', 'Feature', 'Coefficient', 'Odds Ratio', 'P-Value', 'Effect']]

    return results_df

def get_auc_roc(fit, X, y_true, party_mapping):
    """
    Calculates macro, weighted, and per-class OvR AUC ROC scores.
    """

    y_pred_probs = fit.predict(X)

    auc_macro = roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro")
    auc_weighted = roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="weighted")
    
    print(f"One-vs-Rest ROC AUC (Macro Average): {auc_macro:.4f}")
    print(f"One-vs-Rest ROC AUC (Weighted Average): {auc_weighted:.4f}")
    print("---")

    # 3. Calculate per-class AUC
    print("--- ROC AUC Score per Party (One-vs-Rest) ---")
    performace_data = []

    if not hasattr(y_pred_probs, 'iloc'):
        y_pred_probs = pd.DataFrame(y_pred_probs)

    for party_code, party_name in party_mapping.items():
        
        y_true_class = (y_true == party_code).astype(int)
        y_pred_probs_class = y_pred_probs.iloc[:, party_code]
        
        auc_class = roc_auc_score(y_true_class, y_pred_probs_class)
        
        print(f"{party_name} (Class {party_code}): {auc_class:.4f}")
        performace_data.append({
            'party': party_name,
            'code': party_code,
            'auc': auc_class
        })
        
    return performace_data

def evaluate_prediction_model(df_party, X, party_mapping, fit, top_head=3):
    party_names = list(party_mapping.values())

    feature_names = X.columns.drop('const')
    
    y_true = df_party['party_code']

    predicted_probs = fit.predict(X)
    y_pred = np.argmax(predicted_probs, axis=1)

    top_odds_ratios = get_top_significant_odds_ratios(fit, party_mapping, top_head=top_head)
    # plot_confusion_matrix(y_true, y_pred, party_names)
    
    return get_auc_roc(fit, X, y_true, party_mapping), top_odds_ratios

def get_actual_pred(df_party, X, party_mapping, fit, top_head=3):
    party_names = list(party_mapping.values())

    feature_names = X.columns.drop('const')

    y_true = df_party['party_code']

    predicted_probs = fit.predict(X)
    y_pred = np.argmax(predicted_probs, axis=1)

    top_odds_ratios = get_top_significant_odds_ratios(fit, party_mapping, top_head=top_head)
    return y_true, y_pred, party_names, top_odds_ratios