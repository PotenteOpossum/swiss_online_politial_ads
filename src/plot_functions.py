import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast, os
from scipy.stats import wasserstein_distance
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from datetime import timedelta
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import ConnectionPatch
from sklearn.metrics import confusion_matrix

def create_demographic_plot(df, parties, party_colors, output_dir='imgs', save=False):
    """
    Generates and displays a set of demographic distribution plots for political parties.
    
    This function creates a grid of plots comparing the gender and age distribution of impressions
    for specified political parties against the complementary set of parties.
    """
    # Set the global theme for all plots in this function
    sns.set_theme(style="ticks", context="paper", font_scale=1.85)
    
    # --- Data Preprocessing ---
    def safe_eval(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                 return []
        return val

    df['demographic_distribution'] = df['demographic_distribution'].apply(safe_eval)
    df['impressions'] = (df['impressions_lower'] + df['impressions_upper']) / 2

    # Explode and flatten the demographic data
    df_exploded = df.explode('demographic_distribution').reset_index(drop=True)
    demo_df = pd.json_normalize(df_exploded['demographic_distribution']).reset_index(drop=True)
    df_flat = pd.concat([df_exploded[['party_name', 'impressions']], demo_df], axis=1)

    # Clean and calculate impressions per demographic group
    df_flat['percentage'] = pd.to_numeric(df_flat['percentage'], errors='coerce')
    df_flat.dropna(subset=['percentage'], inplace=True)
    df_flat['demo_impressions'] = df_flat['impressions'] * df_flat['percentage']
    df_flat['gender'] = df_flat['gender'].str.capitalize()

    # Define ordering for age groups
    age_order = ['13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    age_midpoints = [15, 21, 29.5, 39.5, 49.5, 59.5, 70] # For Wasserstein distance
    
    # --- Plotting Setup ---
    n_parties = len(parties)
    # fig, axes = plt.subplots(2, n_parties, figsize=(n_parties * 3.5, 9), sharey='row', constrained_layout=True)
    n_parties = len(parties)
    # Add gridspec_kw={'hspace': 0.4} to control the vertical spacing
    fig, axes = plt.subplots(2, n_parties, figsize=(n_parties * 3.5, 9), sharey='row', 
                             constrained_layout=True, gridspec_kw={'hspace': 0.05})

    complementary_color = '#C8B9A6'

    # --- Generate Plots for Each Party ---
    for i, party in enumerate(parties):
        ax_gender = axes[0, i]
        ax_age = axes[1, i]

        # Filter data for the current party and its complement
        tagged_ads = df[df['party_name'] == party]
        tagged_flat = df_flat[df_flat['party_name'] == party]
        complementary_flat = df_flat[df_flat['party_name'] != party]

        n_ads_tagged = len(tagged_ads)
        total_imps_tagged = tagged_ads['impressions'].sum()

        # --- Gender Plot ---
        tagged_gender_dist = tagged_flat.groupby('gender')['demo_impressions'].sum()
        if tagged_gender_dist.sum() > 0:
            tagged_gender_dist /= tagged_gender_dist.sum()

        comp_gender_dist = complementary_flat.groupby('gender')['demo_impressions'].sum()
        if comp_gender_dist.sum() > 0:
            comp_gender_dist /= comp_gender_dist.sum()
        
        # Calculate Odds Ratio
        tagged_male_imps = tagged_flat[tagged_flat['gender'] == 'Male']['demo_impressions'].sum()
        tagged_female_imps = tagged_flat[tagged_flat['gender'] == 'Female']['demo_impressions'].sum()
        comp_male_imps = complementary_flat[complementary_flat['gender'] == 'Male']['demo_impressions'].sum()
        comp_female_imps = complementary_flat[complementary_flat['gender'] == 'Female']['demo_impressions'].sum()

        odds_ratio = np.nan
        if tagged_female_imps > 0 and comp_female_imps > 0 and comp_male_imps > 0:
            odds_ratio = (tagged_male_imps / tagged_female_imps) / (comp_male_imps / comp_female_imps)

        gender_labels = ['Male', 'Female']
        x_gender = np.arange(len(gender_labels))
        width = 0.4

        rects1 = ax_gender.bar(x_gender - width/2, [tagged_gender_dist.get(g, 0) for g in gender_labels], width, color=party_colors.get(party, 'blue'), zorder=3)
        rects2 = ax_gender.bar(x_gender + width/2, [comp_gender_dist.get(g, 0) for g in gender_labels], width, color=complementary_color, zorder=3)
        
        ax_gender.set_title(party, color=party_colors.get(party, 'blue'), fontweight='bold', fontsize='large', pad=20)
        ax_gender.set_xticks(x_gender)
        ax_gender.set_xticklabels(gender_labels)
        ax_gender.grid(axis='y', linestyle=':', linewidth=0.7, zorder=0)
        ax_gender.tick_params(axis='x', length=0)
        ax_gender.text(0.95, 1.05, f"OR: {odds_ratio:.2f}", transform=ax_gender.transAxes, ha='right', va='top', fontsize='small')#fontweight='bold',
        # ax_gender.set_xlabel(f"{n_ads_tagged} ads, {total_imps_tagged/1e6:.2f}M imps", color='gray', fontsize='small', labelpad=20)
        # ax_gender.text(0.95, 1.05, f"{n_ads_tagged} ads, {total_imps_tagged/1e6:.2f}M imps", transform=ax_gender.transAxes, ha='right', va='top', fontsize='small')

        for rect in rects1 + rects2:
            height = rect.get_height()
            if height > 0:
                ax_gender.annotate(f'{height:.2f}'.lstrip('0'), xy=(rect.get_x() + rect.get_width() / 2, height),
                                   xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize='small')
        
        # --- Age Plot ---
        tagged_age_dist = tagged_flat.groupby('age')['demo_impressions'].sum().reindex(age_order, fill_value=0)
        if tagged_age_dist.sum() > 0:
            tagged_age_dist /= tagged_age_dist.sum()

        comp_age_dist = complementary_flat.groupby('age')['demo_impressions'].sum().reindex(age_order, fill_value=0)
        if comp_age_dist.sum() > 0:
            comp_age_dist /= comp_age_dist.sum()

        # Calculate Wasserstein Distance
        wasserstein = wasserstein_distance(u_values=age_midpoints, v_values=age_midpoints, u_weights=tagged_age_dist.values, v_weights=comp_age_dist.values)

        x_age = np.arange(len(age_order))
        ax_age.bar(x_age - width/2, tagged_age_dist, width, color=party_colors.get(party, 'blue'), zorder=3)
        ax_age.bar(x_age + width/2, comp_age_dist, width, color=complementary_color, zorder=3)
        
        ax_age.set_xticks(x_age)
        ax_age.set_xticklabels(age_order, rotation=45)
        ax_age.grid(axis='y', linestyle=':', linewidth=0.7, zorder=0)
        ax_age.tick_params(axis='x', length=0)
        ax_age.text(0.68, 0.95, f"WD: {wasserstein:.2f}", transform=ax_age.transAxes, ha='left', va='top', fontsize='small')#fontweight='bold',

    # --- Final Figure-Level Adjustments ---
    axes[0, 0].set_ylabel('Gender Distribution')
    axes[1, 0].set_ylabel('Age Distribution')
    
    # Remove spines for a cleaner look
    for ax_row in axes:
        for ax in ax_row:
            sns.despine(ax=ax, left=True, bottom=True)

    # --- Add custom legend ---
    # Handle for the colored patch
    legend_patch = mpatches.Patch(color=complementary_color, label='Party Complementary')
    # Create invisible handles for the text-only legend entries
    wd_handle = mpatches.Patch(alpha=0, label='WD = Wasserstein distance')
    or_handle = mpatches.Patch(alpha=0, label='OR = Odds ratio')

    fig.legend(
        handles=[legend_patch, wd_handle, or_handle], 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.08),  # Adjusted y-position for better spacing
        ncol=3,                      # Arrange legend items in 3 columns
        frameon=False
    )

    if save:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        plot_filepath = os.path.join(output_dir, 'demographic_distribution.pdf')
        plt.savefig(plot_filepath, bbox_inches='tight')
        print(f"Time series plot saved to '{plot_filepath}'")

    plt.show()

def create_gender_odds_ratio_heatmap(df, parties, party_colors,languages, n_bootstrap=100, output_dir='imgs', save=False):

    def safe_eval(val):
        if isinstance(val, str):
            return ast.literal_eval(val)
        return val

    # CHANGED: Replaced set_context with set_theme and font_scale
    sns.set_theme(style="ticks", context="paper", font_scale=1.85)
    
    # --- Data Preprocessing ---
    df = df.copy()

    df['lang'] = df['lang'].map({
        'italian': 'IT',
        'french': 'FR',
        'german': 'DE'
    })
    
    if 'index' in df.columns or 'level_0' in df.columns:
         df = df.reset_index().rename(columns={'index': 'ad_id', 'level_0': 'ad_id'})
    elif 'ad_id' not in df.columns:
         df = df.reset_index().rename(columns={'index': 'ad_id'})
         
    df['demographic_distribution'] = df['demographic_distribution'].apply(safe_eval)
    df['impressions'] = (df['impressions_lower'] + df['impressions_upper']) / 2

    df_exploded = df.explode('demographic_distribution')
    demo_df = pd.json_normalize(df_exploded['demographic_distribution']).reset_index(drop=True)
    df_exploded.reset_index(drop=True, inplace=True)
    df_flat = pd.concat([df_exploded[['ad_id', 'party_name', 'impressions', 'lang']], demo_df], axis=1)

    df_flat['percentage'] = pd.to_numeric(df_flat['percentage'], errors='coerce')
    df_flat.dropna(subset=['percentage'], inplace=True)
    df_flat['demo_impressions'] = df_flat['impressions'] * df_flat['percentage']
    df_flat['gender'] = df_flat['gender'].str.capitalize()
    
    # --- Bootstrapping Loop ---
    odds_ratios = pd.DataFrame(index=languages, columns=parties, dtype=float)
    annot_text = pd.DataFrame(index=languages, columns=parties, dtype=object)

    for lang in languages:
        lang_ads = df[df['lang'] == lang]
        lang_flat = df_flat[df_flat['lang'] == lang] 
        
        for party in parties:
            tagged_ads = lang_ads[lang_ads['party_name'] == party]
            complementary_ads = lang_ads[lang_ads['party_name'] != party]
            n_ads_tagged = len(tagged_ads)

            if n_ads_tagged < 10 or len(complementary_ads) == 0:
                odds_ratios.loc[lang, party] = np.nan
                annot_text.loc[lang, party] = "n/a"
                continue

            # --- Start Bootstrap ---
            bootstrap_ors = []
            for _ in range(n_bootstrap):
                sample_tagged_ids = tagged_ads.sample(n=n_ads_tagged, replace=True)['ad_id']
                sample_comp_ids = complementary_ads.sample(n=len(complementary_ads), replace=True)['ad_id']
                
                tagged_flat_sample = lang_flat[lang_flat['ad_id'].isin(sample_tagged_ids)]
                comp_flat_sample = lang_flat[lang_flat['ad_id'].isin(sample_comp_ids)]

                tagged_male_imps = tagged_flat_sample[tagged_flat_sample['gender'] == 'Male']['demo_impressions'].sum()
                tagged_female_imps = tagged_flat_sample[tagged_flat_sample['gender'] == 'Female']['demo_impressions'].sum()
                comp_male_imps = comp_flat_sample[comp_flat_sample['gender'] == 'Male']['demo_impressions'].sum()
                comp_female_imps = comp_flat_sample[comp_flat_sample['gender'] == 'Female']['demo_impressions'].sum()

                if tagged_female_imps > 0 and comp_female_imps > 0 and comp_male_imps > 0:
                    odds_ratio = (tagged_male_imps / tagged_female_imps) / (comp_male_imps / comp_female_imps)
                    bootstrap_ors.append(odds_ratio)
                else:
                    bootstrap_ors.append(np.nan)
            # --- End Bootstrap ---

            mean_or = np.nanmean(bootstrap_ors)
            ci_low = np.nanpercentile(bootstrap_ors, 2.5)
            ci_high = np.nanpercentile(bootstrap_ors, 97.5)
            
            odds_ratios.loc[lang, party] = mean_or
            
            text = f"{mean_or:.2f}"
            if not (ci_low <= 1.0 <= ci_high):
                text += "*"
            annot_text.loc[lang, party] = text

    lang_colors = {'DE': 'black', 'FR': 'black', 'IT': 'black'}

    plt.figure(figsize=(len(parties) * 1.8, len(languages) * 1.8))
    ax = sns.heatmap(
        odds_ratios,
        annot=annot_text,
        fmt="",
        cmap=sns.color_palette("vlag", as_cmap=True),
        center=1.0,
        linewidths=2,
        linecolor='white',
        cbar=True,
        # CHANGED: Removed "size"
        annot_kws={"weight": "bold"}
    )
    ax.set_facecolor('#d8dcd6')
    
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    
    for tick_label in ax.get_xticklabels():
        tick_label.set_color(party_colors.get(tick_label.get_text(), 'black'))
        tick_label.set_fontweight('bold')
        # CHANGED: Removed set_fontsize
    
    for tick_label in ax.get_yticklabels():
        tick_label.set_color(lang_colors.get(tick_label.get_text(), 'black'))
        tick_label.set_fontweight('bold')
        # CHANGED: Removed set_fontsize
        tick_label.set_rotation(0)

    cbar = ax.collections[0].colorbar
    cbar.set_label('Male-to-Female Odds Ratio', labelpad=15)

    # CHANGED: Removed fontsize
    # plt.figtext(0.5, -0.05, "* 95% confidence interval does not include 1.0", 
    #             ha="center", style='italic', 
    #             bbox={"facecolor":"white", "alpha":0, "pad":5})
    if save:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        plot_filepath = os.path.join(output_dir, 'gender_distribution.pdf')
        plt.savefig(plot_filepath, bbox_inches='tight')
        print(f"Time series plot saved to '{plot_filepath}'")

    plt.show()

def create_language_specific_age_plot(df, parties, party_colors, languages, n_bootstrap=100, output_dir='imgs', save=False):
    """
    Generates language-specific age distribution plots for political parties,
    styled to match the demographic plot theme.
    """
    # Style update: Set the global theme for all plots in this function
    sns.set_theme(style="ticks", context="paper", font_scale=2)

    # --- Data Preprocessing ---
    df = df.copy()

    df['lang'] = df['lang'].map({
        'italian': 'IT',
        'french': 'FR',
        'german': 'DE'
    })

    def safe_eval(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                 return []
        return val

    df = df.reset_index().rename(columns={'index': 'ad_id'})
    df['demographic_distribution'] = df['demographic_distribution'].apply(safe_eval)
    df['impressions'] = (df['impressions_lower'] + df['impressions_upper']) / 2

    df_exploded = df.explode('demographic_distribution')
    demo_df = pd.json_normalize(df_exploded['demographic_distribution']).reset_index(drop=True)
    df_exploded.reset_index(drop=True, inplace=True)
    df_flat = pd.concat([df_exploded[['ad_id', 'party_name', 'impressions', 'lang']], demo_df], axis=1)

    df_flat['percentage'] = pd.to_numeric(df_flat['percentage'], errors='coerce')
    df_flat.dropna(subset=['percentage'], inplace=True)
    df_flat['demo_impressions'] = df_flat['impressions'] * df_flat['percentage']

    age_order = ['13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    age_midpoints = [15, 21, 29.5, 39.5, 49.5, 59.5, 70]
    
    complementary_color = '#C8B9A6' # Using a similar gray/brown for complement

    # --- Plotting Setup ---
    n_langs = len(languages)
    n_parties = len(parties)
    # Style update: Added gridspec_kw for spacing control
    fig, axes = plt.subplots(n_langs, n_parties, figsize=(n_parties * 4, n_langs * 3.5), 
                             sharex=True, sharey=True, constrained_layout=True,
                             gridspec_kw={'hspace': 0.05, 'wspace': 0.0})

    # Handle cases where axes is not a 2D array
    if n_langs == 1 and n_parties == 1: axes = np.array([[axes]])
    elif n_langs == 1: axes = np.array([axes])
    elif n_parties == 1: axes = np.array([[ax] for ax in axes])
    
    for r, lang in enumerate(languages):
        for c, party in enumerate(parties):
            ax = axes[r, c]

            if lang != 'All':
                tagged_ads = df[(df['lang'] == lang) & (df['party_name'] == party)]
                complementary_ads = df[(df['lang'] == lang) & (df['party_name'] != party)]
            else:
                tagged_ads = df[(df['party_name'] == party)]
                complementary_ads = df[(df['party_name'] != party)]

            total_imps_tagged = tagged_ads['impressions'].sum()
            n_ads_tagged = len(tagged_ads)

            if n_ads_tagged < 10:
                ax.text(0.5, 0.5, "Not enough ads", ha='center', va='center', transform=ax.transAxes, fontsize='small', color='gray')
            else:
                bootstrap_distributions = []
                for _ in range(n_bootstrap):
                    sample_ads = tagged_ads.sample(n=n_ads_tagged, replace=True)
                    sample_flat = df_flat[df_flat['ad_id'].isin(sample_ads['ad_id'])]
                    age_dist = sample_flat.groupby('age')['demo_impressions'].sum().reindex(age_order, fill_value=0)
                    if age_dist.sum() > 0:
                        age_dist /= age_dist.sum()
                    bootstrap_distributions.append(age_dist.values)
                
                bootstrap_distributions = np.array(bootstrap_distributions)
                mean_dist = bootstrap_distributions.mean(axis=0)
                
                bootstrap_data = []
                for dist in bootstrap_distributions:
                    for age, value in zip(age_order, dist):
                        bootstrap_data.append({'age': age, 'distribution': value})
                bootstrap_df = pd.DataFrame(bootstrap_data)

                comp_flat = df_flat[df_flat['ad_id'].isin(complementary_ads['ad_id'])]
                comp_age_dist = comp_flat.groupby('age')['demo_impressions'].sum().reindex(age_order, fill_value=0)
                if comp_age_dist.sum() > 0:
                    comp_age_dist /= comp_age_dist.sum()

                wasserstein = wasserstein_distance(u_values=age_midpoints, v_values=age_midpoints,
                                                               u_weights=mean_dist, v_weights=comp_age_dist.values)
                
                party_color = party_colors.get(party, 'blue')
                
                # Plotting complementary distribution as a line
                sns.lineplot(x=age_order, y=comp_age_dist.values, ax=ax, color=complementary_color, zorder=1, 
                             marker='s', markersize=4, markerfacecolor='white', markeredgecolor=complementary_color, legend=False)
                             
                # Plotting party distribution with bootstrap confidence interval
                sns.lineplot(data=bootstrap_df, x='age', y='distribution', ax=ax, color=party_color, zorder=3, 
                             marker='.', markersize=8, errorbar='sd', err_kws={'alpha': 0.2}, legend=False)

                # Style update: Using relative font sizes
                ax.text(0.65, 0.95, f"WD: {wasserstein:.2f}", transform=ax.transAxes, ha='left', va='top', fontsize='medium')
            
            # --- Axis Styling ---
            # Style update: Match grid and despine style from previous plot
            ax.grid(axis='x', linestyle=':', linewidth=0.81, zorder=0)
            sns.despine(ax=ax, left=True, bottom=True)
            ax.tick_params(axis='x', length=0)
            # ax.set_yticks(np.arange(0, 0.5, 0.1))

            # --- Labels and Titles ---
            if r == 0:
                # Style update: Consistent title styling with padding
                ax.set_title(party, color=party_colors.get(party, 'blue'), fontweight='bold', fontsize='large', pad=20)
            if c == 0:
                ax.set_ylabel(lang.upper(), fontweight='bold')
            if r == n_langs - 1:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                ax.tick_params(axis='x', labelbottom=False)

    # --- Final Figure-Level Adjustments ---
    comp_handle = mlines.Line2D([], [], color=complementary_color, marker='s', markersize=5, mfc='white', linestyle='None', label='Party Complementary')
    text_handle = plt.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', linewidth=0, label='WD = Wasserstein Distance')
    
    # Style update: Let font_scale manage legend font size
    fig.legend(handles=[comp_handle, text_handle], loc='lower center', bbox_to_anchor=(0.5, -0.075), ncol=2, frameon=False)

    if save:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        plot_filepath = os.path.join(output_dir, 'age_distribution.pdf')
        plt.savefig(plot_filepath, bbox_inches='tight')
        print(f"Time series plot saved to '{plot_filepath}'")
    
    plt.show()

def human_readable_formatter(val, pos):
    """Converts a large number into a human-readable string with K/M."""
    if val >= 1_000_000:
        return f'{val / 1_000_000:g}M'
    elif val >= 1_000:
        return f'{val / 1_000:g}K'
    else:
        return f'{val:g}'

def plot_impressions_time_series(df, output_dir='imgs', start_date=pd.to_datetime('2021-01-01', utc=True), end_date=pd.to_datetime('2025-10-01', utc=True), save=False, event_dates=None, show_plot=True, figsize=(18, 8), inset_pos_size=[0.17, 0.45, 0.35, 0.5]):
    """
    Analyzes and plots time series data for total impressions. The main plot
    shows a smoothed 7-day rolling average of the aggregated total, while an 
    inset plot provides a daily zoom-in on the month before the federal 
    election, broken down by language.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least ['date', 'lang', 'impressions_avg'] columns.
    output_dir : str
        Directory to save the plot in.
    start_date : str or datetime, optional
        Start date for filtering the data.
    end_date : str or datetime, optional
        End date for filtering the data.
    save : bool
        If True, saves the plot to a file.
    event_dates : list of str or datetime, optional
        A list of dates to mark with vertical lines.
    show_plot : bool
        If True, displays the plot.
    figsize : tuple
        The size of the figure.
    """

    # --- Seaborn Styling ---
    sns.set_theme(style="ticks", context="paper", font_scale=2.7)
    y_max = 6000000#5814416

    # --- Data Filtering ---
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date, utc=True)]
    
    if df.empty:
        print(f"No data found for the specified time range: {start_date} to {end_date}")
        return

    # --- Data Aggregation for Main Plot (Total Impressions) ---
    main_daily_data = df.groupby(['date']).agg({
        'impressions_avg': 'sum'
    }).reset_index()
    # Add a 7-day rolling average for a smoother line
    main_daily_data['impressions_avg_smoothed'] = main_daily_data['impressions_avg'].rolling(window=7, center=True, min_periods=1).mean()
    print(main_daily_data['impressions_avg_smoothed'].max())

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use distinct colors from the reference image
    colors = {'italian': '#C44E52', 'german': '#DD8452', 'french': '#55A868', 'other': 'grey'}
    
    # --- Main Impressions Plot (Smoothed) ---
    sns.lineplot(
        data=main_daily_data,
        x='date',
        # y='impressions_avg',
        y='impressions_avg_smoothed',
        ax=ax,
        label='Total Impressions (7-Day Avg)',
        color='#4C72B0'
    )

    # Get y-limits after plotting the main line for robust scaling
    plot_ymin, plot_ymax = ax.get_ylim()
    plot_ymax = y_max
    # --- Plot Vertical Bars for Event Dates ---
    if event_dates is not None and len(event_dates) > 0:
        event_dates_dt = pd.to_datetime(event_dates, utc=True)

        filtered_events = event_dates_dt[
             (event_dates_dt >= df['date'].min()) & 
             (event_dates_dt <= df['date'].max())
        ]
        
        federal_election_date = pd.to_datetime('2023-10-22', utc=True)
        referendum_labeled = False
        election_labeled = False

        for event_date in filtered_events:
            label = "_nolegend_"
            if event_date == federal_election_date:
                if not election_labeled:
                    label = 'Federal Election'
                    election_labeled = True
                ax.vlines(x=event_date, ymin=0, ymax=plot_ymax, color='tomato', linewidth=1.7, linestyle='-.', alpha=0.7, label=label)
            else:
                if not referendum_labeled:
                    label = 'Referendum'
                    referendum_labeled = True
                ax.vlines(x=event_date, ymin=0, ymax=plot_ymax, color='gray', linewidth=1.7, linestyle='-', alpha=0.5, label=label)

    # --- Zoom-in Inset Plot (Grouped by Language) ---
    federal_election_date = pd.to_datetime('2023-10-22', utc=True)
    zoom_start = federal_election_date - pd.Timedelta(days=30)
    zoom_end = federal_election_date + pd.Timedelta(days=1)

    axins = ax.inset_axes(inset_pos_size, zorder=10)
    
    lang_daily_data = df.groupby(['date', 'lang']).agg({'impressions_avg': 'sum'}).reset_index()
    lang_daily_data['lang'] = lang_daily_data['lang'].str.lower()

    # sns.lineplot(data=lang_daily_data, x='date', y='impressions_avg', hue='lang', markers=['o', '<', '>'], palette=colors, ax=axins)
    marker_map = {'italian': '>', 'german': 'o', 'french': '<'}
    sns.lineplot(
        data=lang_daily_data, 
        x='date', 
        y='impressions_avg', 
        hue='lang', 
        style='lang',  # Add this
        markers=marker_map,  # Add this
        palette=colors,
        markersize=6,
        ax=axins
    )

    lang_handles, lang_labels = axins.get_legend_handles_labels()
    axins.get_legend().remove()
        
    axins.set_xlim(zoom_start, zoom_end)
    
    zoom_data = lang_daily_data[(lang_daily_data['date'] >= zoom_start) & (lang_daily_data['date'] <= zoom_end)]
    if not zoom_data.empty:
        y1, y2 = zoom_data['impressions_avg'].min(), zoom_data['impressions_avg'].max()
        # axins.set_ylim(y1 - (y2-y1)*0.1, y2 + (y2-y1)*0.1)
        print('limit!')
        print(y1)
        print(y2)
        axins.set_ylim(y1 - (y2-y1)*0.1, y2+y2*0.05)

    if 'filtered_events' in locals():
        inset_ymax = 8000000#axins.get_ylim()[1]
        for event_date in filtered_events:
            if zoom_start <= event_date <= zoom_end:
                 if event_date == federal_election_date:
                     axins.vlines(x=event_date, ymin=0, ymax=y2+y2*0.05, color='tomato', linewidth=1.7, linestyle='-.', alpha=0.7)
                 else:
                     axins.vlines(x=event_date, ymin=0, ymax=inset_ymax, color='gray', linewidth=1.7, linestyle='-', alpha=0.7)

    axins.set_xlabel('')
    axins.set_ylabel('')
    
    tick_dates = [zoom_start.to_pydatetime(), 
                  (zoom_start + timedelta(days=10)).to_pydatetime(),
                  (zoom_start + timedelta(days=20)).to_pydatetime(),
                  federal_election_date.to_pydatetime()]
    tick_dates = sorted(list(set(tick_dates)))
    axins.set_xticks(tick_dates)

    axins.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axins.tick_params(axis='x', rotation=30)
    axins.tick_params(axis='y', labelleft=True)
    axins.yaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.7)

    # --- Shaded area on main plot for zoom region ---
    shade_start = federal_election_date - pd.Timedelta(days=30)
    shade_end = federal_election_date
    ax.fill_between([shade_start, shade_end], 0, plot_ymax, color='gray', alpha=0.2, zorder=0)
    ax.set_ylim(plot_ymin, plot_ymax) # Restore y-limits after fill_between

    current_ticks = ax.get_xticks()
    ax.set_xlim(main_daily_data['date'].min() - pd.Timedelta(days=3), main_daily_data['date'].max() + pd.Timedelta(days=3))
    ax.set_xticks(current_ticks[1:])

    yticks = [0, 2000000, 4000000, 6000000]
    ax.set_yticks(yticks)
    # ax.set_ylim(0, 6000000)

    yticks = [0, 4000000, 8000000]
    axins.set_yticks(yticks)
    # axins.set_ylim(0, 8000000)

    # --- Connector Lines ---
    shade_start_num = mdates.date2num(shade_start)
    con_top = ConnectionPatch(xyA=(1, 1), xyB=(shade_start_num, plot_ymax),
                              coordsA='axes fraction', coordsB='data',
                              axesA=axins, axesB=ax, alpha=.42,
                              color='gray', linestyle='-', linewidth=1)
    con_bottom = ConnectionPatch(xyA=(1, 0), xyB=(shade_start_num, 0),
                                 coordsA='axes fraction', coordsB='data',
                                 axesA=axins, axesB=ax, alpha=.42,
                                 color='gray', linestyle='-', linewidth=1)
    fig.add_artist(con_top)
    fig.add_artist(con_bottom)

    # --- Final Touches ---
    ax.set_ylabel('Impressions')
    ax.set_xlabel('Date')
    ax.yaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.7)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(human_readable_formatter))
    axins.yaxis.set_major_formatter(mticker.FuncFormatter(human_readable_formatter))

    plt.setp(ax.get_xticklabels(), rotation=30, ha="center")
    
    # --- Merged Legend ---
    main_handles, main_labels = ax.get_legend_handles_labels()
    capitalized_lang_labels = [l.capitalize() for l in lang_labels]
    all_handles = main_handles + lang_handles
    all_labels = main_labels + capitalized_lang_labels
    
    if ax.get_legend() is not None: ax.get_legend().remove()
    fig.legend(handles=all_handles, labels=all_labels, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=int(len(all_handles)/2), markerscale=2)
    
    sns.despine(ax=ax, left=True, bottom=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        plot_filepath = os.path.join(output_dir, 'timeseries_impressions.pdf')
        plt.savefig(plot_filepath, bbox_inches='tight')
        print(f"Time series plot saved to '{plot_filepath}'")
        
    if show_plot:
        plt.show()
    else:
        plt.close()
    
def plot_gam_subplots(spending_results, impressions_results, window_days=30, output_dir='imgs', save=False):
    print(f"\n--- Generating 2x2 Combined Subplots ---")
    
    # Set seaborn theme
    sns.set_theme(style="ticks", context="paper", font_scale=2.55)
    
    # Define the 2x2 grid, SHARING axes
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 14), 
                           sharex='col', sharey='row')
    
    # Make plots closer
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    lang_names = ['italian', 'german', 'french', 'all']
    plot_titles = {
        'italian': 'Italian',
        'german': 'German',
        'french': 'French',
        'all': 'All Languages'
    }
    
    # Colors for the two lines
    color_spending = '#0072B2' # Blue
    color_impressions = '#D55E00' # Vermillion/Red
    
    # Map languages to their specific subplot
    axes_map = {
        'italian': axes[0, 0],
        'german':  axes[0, 1],
        'french':  axes[1, 0],
        'all':     axes[1, 1]
    }
    
    # Loop using the map
    for lang_name, ax in axes_map.items():
        plot_title = plot_titles.get(lang_name, lang_name)
        
        # Add text label in the TOP RIGHT corner
        ax.text(0.95, 0.95, plot_title, transform=ax.transAxes, 
                ha='right', va='top', fontweight='bold')
        
        has_data = False
        
        # Plot Spending
        if lang_name in spending_results:
            x_s, y_s, conf_s = spending_results[lang_name]
            if x_s is not None:
                ax.plot(x_s, y_s, color=color_spending, label='Expenditure', linewidth=2.5)
                ax.fill_between(x_s, conf_s[:, 0], conf_s[:, 1], color=color_spending, alpha=0.11)
                has_data = True
        
        # Plot Impressions
        if lang_name in impressions_results:
            x_i, y_i, conf_i = impressions_results[lang_name]
            if x_i is not None:
                ax.plot(x_i, y_i, color=color_impressions, label='Impressions', linewidth=2.5)
                ax.fill_between(x_i, conf_i[:, 0], conf_i[:, 1], color=color_impressions, alpha=0.11)
                has_data = True

        # Add common elements
        if has_data:
            ax.axvline(0, color='red', linestyle='--', label='Referendum Day (t=0)')
            ax.axhline(0, color='black', linestyle=':', linewidth=1, label='Baseline')
            ax.set_xlim(-window_days, window_days)
            
            # Set vertical grid lines only
            ax.grid(True, axis='x', linestyle=':', color='#CCCCCC', alpha=0.7)
            
            # Set a reduced number of integer ticks for the x-axis
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=7, integer=True))

        else:
            ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='gray')

        # --- Add individual axis labels to outer plots ---
        # Add X label to the bottom row
        if ax in [axes[1, 0], axes[1, 1]]:
            ax.set_xlabel('Days to Nearest Referendum')
            
        # Add Y label to the left column
        if ax in [axes[0, 0], axes[1, 0]]:
            ax.set_ylabel('Partial Effect') # 'Partial Effect (on log(y+1) scale)'


    # --- Manually set Y-Ticks ---
    # Set the desired Y-ticks
    yticks = [4, 2, 0, -2, -4]
    
    # Apply the manual ticks to the left-most plots (which control each row)
    axes[0, 0].set_yticks(yticks)
    axes[1, 0].set_yticks(yticks)
    
    # Manually set the Y-limits to ensure ticks are displayed well
    axes[0, 0].set_ylim(-5.2, 5.2)
    axes[1, 0].set_ylim(-5.2, 5.2)

    xticks = [30, 15, 0, -15, -30]

    axes[1, 0].set_xticks(xticks)
    axes[1, 1].set_xticks(xticks)

    # Remove all spines (borders)
    sns.despine(fig=fig, left=True, bottom=True)
    
    # --- Add Legend to the TOP ---
    handles, labels = axes[0,0].get_legend_handles_labels() 
    fig.legend(handles=handles, labels=labels, 
               loc='upper center', 
               bbox_to_anchor=(0.55, .98), # Position at the top
               ncol=4, 
               frameon=False) 

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    if save:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        plot_filepath = os.path.join(output_dir, 'GAM_model.pdf')
        plt.savefig(plot_filepath, bbox_inches='tight')
        print(f"Time series plot saved to '{plot_filepath}'")

    plt.show()

def plot_boxplot(df, output_dir='imgs', save=True):

# --- 1. Set the Theme ---
    sns.set_theme(style="ticks", context="paper", font_scale=2.1)

    # --- 2. Data Preprocessing ---
    df = df.dropna(subset=['date'])

    df = df.rename(columns={
        'Impressions no': 'Opposing',
        'Impressions yes': 'Supporting'
    })
    df['Supporting'] = pd.to_numeric(df['Supporting'], errors='coerce')
    df['Opposing'] = pd.to_numeric(df['Opposing'], errors='coerce')
    
    df_melted = df.melt(id_vars=['approved'], 
                        value_vars=['Supporting', 'Opposing'],
                        var_name='Campaign Side', 
                        value_name='Impressions')
    
    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.boxplot(x='approved', 
                y='Impressions', 
                hue='Campaign Side', 
                data=df_melted, 
                ax=ax,
                palette={'Supporting': 'skyblue', 'Opposing': 'salmon'},
                zorder=3)
    
    # --- 4. Formatting the Plot ---
    ax.set_xlabel('Referendum Outcome')
    ax.set_ylabel('Total Impressions')
    
    ax.grid(axis='y', linestyle=':', linewidth=0.7, zorder=0)
    
    # --- 3. Apply the new formatter ---
    # Replace the old ticklabel_format line with this:
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(human_readable_formatter))
    
    plt.xticks(rotation=0) 
    
    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(axis='x', length=0) 
    
    plt.legend(title='Campaign Side', frameon=False) 
    plt.tight_layout()

    if save:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        plot_filepath = os.path.join(output_dir, 'outcome_vs_impressions.pdf')
        plt.savefig(plot_filepath, bbox_inches='tight')
        print(f"Time series plot saved to '{plot_filepath}'")
    plt.show()

def plot_party_topic_share_heatmap(df, parties, topics, feature_groups={}, normalize='columns', output_dir='imgs', figsize=(24, 8), xticks = [], save=True):
    """
    Plots a heatmap of party share of voice per topic, with topics grouped
    and group labels displayed at the top.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ad data with 'party_name', 'impressions_avg', and topic columns.
    parties : list
        A list of party names for the heatmap rows, in the desired order.
    topics : list
        A list of topic names (columns in df) for the heatmap columns.
    feature_groups : dict
        Mapping of topic -> group name, used for sorting and labeling the x-axis.
    normalize : str, optional
        'columns' to normalize by topic (share of voice), 'rows' to normalize by party.
    figsize : tuple
        The size of the figure.
    """
    # --- Seaborn Styling ---
    sns.set_theme(style="ticks", context="paper", font_scale=1.9)

    if len(xticks)==0:
        xticks = ['Business &\nRegulation',
        'Economy & Labor',
        'Housing & Rent',
        'Pensions & Retirement',
        'Taxation &\nPublic Finance',
        '',
        'Agriculture &\nFood Security',
        'Climate & Environment',
        'Energy & Policy',
        'Infrastructure & Mobility',
        'Urban & Regional\nDevelopment',
        '',
        'Democratic Process',
        'Foreign Relations',
        'National Security',
        'Political Spectrum',
        '',
        'Civil Liberties & Rights',
        'Culture Society',
        'Education System',
        'Family & Youth Policy',
        'Gender &\nLGBTQ+ Rights',
        'Healthcare System',
        'Immigration & Asylum',
        'Social Justice\n& Equality',
        '',
        'Media & Information']

    if len(feature_groups)==0:
        feature_groups = {
            'Agriculture_Food_Security': 'Environment',
            'Business_Regulation': 'Economy',
            'Civil_Liberties_Rights': 'Society',
            'Climate_Environment': 'Environment',
            'Culture_Society': 'Society',
            'Democratic_Process': 'Governance',
            'Digital_Transformation': 'Technology',
            'Economy_Labor': 'Economy',
            'Education_System': 'Society',
            'Energy_Policy': 'Environment',
            'Family_Youth_Policy': 'Society',
            'Foreign_Relations': 'Governance',
            'Gender_LGBTQ_Rights': 'Society',
            'Governance_Politics': 'Governance',
            'Healthcare_System': 'Society',
            'Housing_Rent': 'Economy',
            'Immigration_Asylum': 'Society',
            'Infrastructure_Mobility': 'Environment',
            'Media_Information': 'Technology',
            'National_Security': 'Governance',
            'Pensions_Retirement': 'Economy',
            'Political_Spectrum': 'Governance',
            'Social_Justice_Equality': 'Society',
            'Taxation_Public_Finance': 'Economy',
            'Urban_Regional_Development': 'Environment',
            'Gender_LGBTQ+_Rights': 'Society',
        }
    
    # --- Sort topics based on feature_groups ---
    temp_df = pd.DataFrame({'Topic': topics})
    temp_df['Group'] = temp_df['Topic'].map(feature_groups).fillna('Other')
    group_order = ['Economy', 'Environment', 'Governance', 'Society', 'Technology', 'Other']
    temp_df['Group'] = pd.Categorical(temp_df['Group'], categories=group_order, ordered=True)
    temp_df.sort_values(by=['Group', 'Topic'], inplace=True)
    sorted_topics = temp_df['Topic'].tolist()

    # --- Prepare Heatmap Data ---
    heatmap_data = pd.DataFrame(index=parties, columns=sorted_topics, data=0.0)
    if normalize == 'columns':   
        for topic in sorted_topics:
            topic_df = df[df[topic] == 1]
            total_impressions_for_topic = topic_df['impressions_avg'].sum()
            if total_impressions_for_topic == 0: continue
            for party in parties:
                party_topic_impressions = topic_df[topic_df['party_name'] == party]['impressions_avg'].sum()
                heatmap_data.loc[party, topic] = party_topic_impressions / total_impressions_for_topic
    elif normalize == 'rows':
        for party in parties:
            party_df = df[df['party_name'] == party]
            total_impressions = party_df['impressions_avg'].sum()
            if total_impressions == 0: continue
            for topic in sorted_topics:
                topic_impressions = party_df[party_df[topic] == 1]['impressions_avg'].sum()
                heatmap_data.loc[party, topic] = (topic_impressions / total_impressions) * 100

    # --- Create a new DataFrame with physical gaps between groups ---
    gapped_data = pd.DataFrame(index=heatmap_data.index)
    last_group = None
    topic_to_group_map = temp_df.set_index('Topic')['Group']
    
    for topic in sorted_topics:
        current_group = topic_to_group_map.get(topic)
        if last_group is not None and current_group != last_group:
            gapped_data[f'_gap_{current_group}'] = np.nan # Insert NaN column
        gapped_data[topic] = heatmap_data[topic]
        last_group = current_group

    # --- Recalculate group info based on the new gapped DataFrame ---
    group_info = {}
    for group_name in group_order:
        group_topics = [col for col in gapped_data.columns if not col.startswith('_gap_') and topic_to_group_map.get(col) == group_name]
        if not group_topics: continue
        
        start_idx = gapped_data.columns.get_loc(group_topics[0])
        end_idx = gapped_data.columns.get_loc(group_topics[-1])
        group_info[group_name] = {'start': start_idx, 'end': end_idx}

    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        gapped_data, # Plot the gapped data
        annot=True,
        fmt=".1f" if normalize == 'rows' else ".2f",
        linewidths=.5,
        ax=ax,
        cmap="Blues",
        cbar_kws={'label': '', 'pad': 0.01},
        cbar=False
    )
    
    # --- Add annotations using the updated group_info ---
    if group_info:
        for group_name, info in group_info.items():
            center_x = info['start'] + (info['end'] - info['start']) / 2 + 0.5
            ax.text(center_x, 1.02, group_name, 
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=18)

    # --- Final Touches ---
    ax.set_xlabel("")
    ax.set_ylabel("")#("Political Party")
    
    # --- Update x-tick labels to hide the gap column names ---
    new_xticklabels = [label if not label.startswith('_gap_') else '' for label in gapped_data.columns]

    print(new_xticklabels)

    if len(xticks)==0:
        new_xticklabels = ['Business &\nRegulation',
        'Economy\n& Labor',
        'Housing\n& Rent',
        'Pensions &\nRetirement',
        'Taxation &\nPublic Finance',
        '',
        'Agriculture &\nFood Security',
        'Climate &\nEnvironment',
        'Energy &\nPolicy',
        'Infrastructure_Mobility',
        'Urban & Regional\nDevelopment',
        '',
        'Democratic\nProcess',
        'Foreign Relations',
        'National Security',
        'Political\nSpectrum',
        '',
        'Civil Liberties\n& Rights',
        'Culture\nSociety',
        'Education\nSystem',
        'Family &\nYouth Policy',
        'Gender &\nLGBTQ+ Rights',
        'Healthcare\nSystem',
        'Immigration_Asylum',
        'Social Justice\n& Equality',
        '',
        'Media &\nInformation']
    else:
        new_xticklabels = xticks

    ax.set_xticklabels(new_xticklabels, rotation=45, ha='right')
    # ax.tick_params(axis='x', length=0) # Hide x-axis tick marks

    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        if gapped_data.columns[i].startswith('_gap_'):
            tick.tick1line.set_visible(False) # Hides the bottom tick line
    
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    party_colors = {
        'FDP': '#cc78bc', 'SP': '#0473b2', 'GRÜNE': '#de8f05',
        'SVP': '#ca9161', 'Die Mitte': '#d55e00', 'GLP': '#009e74'
    }
    for tick_label in ax.get_yticklabels():
        tick_label.set_color(party_colors.get(tick_label.get_text(), 'black'))
        tick_label.set_fontweight('bold')

    sns.despine()#bottom=True, left=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        plot_filepath = os.path.join(output_dir, 'top_topic_correlation.pdf')
        plt.savefig(plot_filepath, bbox_inches='tight')
        print(f"model_coefficients plot saved to '{plot_filepath}'")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, normalization=None, figsize=(10, 8), output_dir='imgs', save=False, file_name=''):
    """
    Generates and plots a confusion matrix with consistent styling.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.
    class_names : list of str
        Display names for all possible classes, in the desired order.
    figsize : tuple, optional
        The size of the figure.
    """
    # --- Seaborn Styling ---
    sns.set_theme(style="ticks", context="paper", font_scale=2.05)
    
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    fmt_string = 'd'
    if normalization == 'recall':
        with np.errstate(divide='ignore', invalid='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)
        fmt_string = '.2f'
    elif normalization == 'precision':
        with np.errstate(divide='ignore', invalid='ignore'):
            cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
            cm = np.nan_to_num(cm)
        fmt_string = '.2f'
    
    # Create a DataFrame for better labeling. The order is already correct.
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    p=['SP', 'GRÜNE', 'GLP', 'Die Mitte', 'FDP', 'SVP']
    cm_df = cm_df.reindex(p)
    cm_df = cm_df.reindex(p, axis=1)
    
    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    # Use fmt='d' for integer display, which is appropriate for counts.
    sns.heatmap(cm_df, annot=True, fmt=fmt_string, cmap='Blues', ax=ax, cbar=False)
    ax.invert_yaxis()
    
    party_colors = {
        'FDP': '#cc78bc', 'SP': '#0473b2', 'GRÜNE': '#de8f05',
        'SVP': '#ca9161', 'Die Mitte': '#d55e00', 'GLP': '#009e74',
    }
    
    for tick_label in ax.get_xticklabels():
        tick_label.set_color(party_colors.get(tick_label.get_text(), 'black'))
        tick_label.set_fontweight('bold')
    
    for tick_label in ax.get_yticklabels():
        tick_label.set_color(party_colors.get(tick_label.get_text(), 'black'))
        tick_label.set_fontweight('bold')
    
    ax.set_ylabel('Actual Party', labelpad=19)
    ax.set_xlabel('Predicted Party', labelpad=19)
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    if save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if file_name=='':
            file_name = 'confusion_matrix.pdf'
        plot_filepath = os.path.join(output_dir, file_name)
        plt.savefig(plot_filepath, bbox_inches='tight')
        print(f"Time series plot saved to '{plot_filepath}'")
    plt.show()

def plot_mnlogit_coefficients_fancy(
    effects_df,
    figsize=(20, 8),
    party_order=['SP', 'GRÜNE', 'GLP', 'Die Mitte', 'FDP', 'SVP'],
    party_markers=None,
    party_colors=None,
    feature_groups=None,
    alpha=0.9,
    shade_by='feature',
    output_dir='./',
    save=False
):
    """
    Plot all features' coefficients across parties using a fancy seaborn style.

    Parameters
    ----------
    effects_df : pd.DataFrame
        A DataFrame with columns: ['Party', 'Feature', 'Coefficient' or 'Marginal Effect'].
    figsize : tuple
        Figure size.
    party_order : list, optional
        A list of party names in the desired order for plotting.
        If None, parties will be sorted alphabetically.
    party_markers : dict, optional
        Mapping from party to marker shape.
    party_colors : dict, optional
        Mapping from party to color.
    feature_groups : dict, optional
        Mapping of feature -> group name, for shaded background bands.
    alpha : float
        Marker transparency.
    shade_by : str, optional
        Controls background shading ('group' or 'feature').
    save : bool
        If True, saves the plot to a file.
    output_dir : str
        Directory to save the plot in.
    """

    if feature_groups==None:
        feature_groups = {
            'Agriculture_Food_Security': 'Environment',
            'Business_Regulation': 'Economy',
            'Civil_Liberties_Rights': 'Society',
            'Climate_Environment': 'Environment',
            'Culture_Society': 'Society',
            'Democratic_Process': 'Governance',
            'Digital_Transformation': 'Technology',
            'Economy_Labor': 'Economy',
            'Education_System': 'Society',
            'Energy_Policy': 'Environment',
            'Family_Youth_Policy': 'Society',
            'Foreign_Relations': 'Governance',
            'Gender_LGBTQ_Rights': 'Society',
            'Governance_Politics': 'Governance',
            'Healthcare_System': 'Society',
            'Housing_Rent': 'Economy',
            'Immigration_Asylum': 'Society',
            'Infrastructure_Mobility': 'Environment',
            'Media_Information': 'Technology',
            'National_Security': 'Governance',
            'Pensions_Retirement': 'Economy',
            'Political_Spectrum': 'Governance',
            'Social_Justice_Equality': 'Society',
            'Taxation_Public_Finance': 'Economy',
            'Urban_Regional_Development': 'Environment'
        }
        feature_groups = {
            k.replace('_', ' '): feature_groups[k] for k in feature_groups.keys()
        }

    # --- Seaborn Styling ---
    sns.set_theme(style="ticks", context="paper", font_scale=1.9)

    df = effects_df.copy()
    df['Feature'] = df['Feature'].str.replace('_', ' ')

    if party_order:
        # Filter the desired order to include only parties present in the data
        parties_in_data = df['Party'].unique()
        parties = [p for p in party_order if p in parties_in_data]
    else:
        # Fallback to alphabetical sorting if party_order is None
        parties = sorted(df['Party'].unique())

    # --- Data Preparation ---
    if feature_groups:
        temp_df = pd.DataFrame({'Feature': df['Feature'].unique()})
        temp_df['Group'] = temp_df['Feature'].map(feature_groups).fillna('Other')
        temp_df.sort_values(by=['Group', 'Feature'], inplace=True)
        features = temp_df['Feature'].tolist()
    else:
        features = sorted(df['Feature'].unique(), key=str.lower)

    # --- Calculate X-coordinates with spacing between groups ---
    x_coords = {}
    tick_positions = []
    group_boundaries_x = []
    current_x = 0
    
    if feature_groups:
        groups = pd.Series(features).map(feature_groups).tolist()
        group_boundaries_x.append(current_x - 0.5)  # Initial boundary
        for i, feature in enumerate(features):
            x_coords[feature] = current_x
            tick_positions.append(current_x)
            if i < len(features) - 1 and groups[i] != groups[i+1]:
                group_boundaries_x.append(current_x + 0.5)
                current_x += 1.75
                group_boundaries_x.append(current_x - 0.5)
            else:
                current_x += 1
        group_boundaries_x.append(current_x - 1 + 0.5) # Final boundary
    else: # No groups
        for i, feature in enumerate(features):
            x_coords[feature] = i
            tick_positions.append(i)
            
    df['x'] = df['Feature'].map(x_coords)

    # --- Add fixed jittering for markers based on the new party order ---
    num_parties = len(parties)
    jitter_width = 0.69
    offsets = np.linspace(-jitter_width / 2, jitter_width / 2, num_parties)
    party_offset_map = dict(zip(parties, offsets))
    df['x'] = df['x'] + df['Party'].map(party_offset_map)

    # --- Default markers and colors (will respect the new party order) ---
    if party_markers is None:
        default_markers = ['^', '>', '<', 'o', 's', 'D', 'v', 'p']
        party_markers = {p: default_markers[i % len(default_markers)] for i, p in enumerate(parties)}
    if party_colors is None:
        palette = sns.color_palette("colorblind", n_colors=len(parties))
        party_colors = {p: palette[i] for i, p in enumerate(parties)}

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- Optional: Shaded background ---
    if feature_groups:
        sorted_groups = pd.Series(features).map(feature_groups).unique().tolist()
        
        if shade_by == 'group':
            for i in range(0, len(group_boundaries_x) - 1, 2):
                if (i // 2) % 2 == 0:
                    start, end = group_boundaries_x[i], group_boundaries_x[i+1]
                    ax.axvspan(start, end, color='grey', alpha=0.1, zorder=0)
        elif shade_by == 'feature':
            for i, pos in enumerate(tick_positions):
                # if i % 2 == 0: # Shade every other feature
                ax.axvspan(pos - 0.42, pos + 0.42, color='grey', alpha=0.1, zorder=0)

        for i in range(0, len(group_boundaries_x) - 1, 2):
            if i//2 < len(sorted_groups):
                start = group_boundaries_x[i]
                group_name = sorted_groups[i // 2]
                ax.text(start, 1.01, group_name, transform=ax.get_xaxis_transform(),
                        ha='left', va='bottom', color='#333')

    # --- Main Scatter Plot using Seaborn ---
    sns.scatterplot(
        data=df,
        x='x',
        y='Marginal Effect',
        hue='Party',
        style='Party',
        hue_order=parties,  # Explicitly set order for hue
        style_order=parties, # Explicitly set order for style
        palette=party_colors,
        markers=party_markers,
        s=99,
        alpha=alpha,
        ax=ax,
        zorder=2,
        legend=False
    )
    # --- Axes Customization and Final Touches ---
    ax.axhline(0, color='grey', linewidth=1, linestyle='-', zorder=1)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.7)

    # Using your provided x_ticks
    x_ticks = [
        'Economy\n& Labor', 'Housing Rent', 'Climate &\nEnvironment', 'Energy Policy',
        'Democratic\nProcess', 'Governance\n& Politics', 'Political\nSpectrum',
        'Civil Liberties\n& Rights', 'Culture &\nSociety', 'Education\nSystem',
        'Healthcare\nSystem', 'Immigration\n& Asylum', 'Social Justice\n& Equality'
    ]

    ax.set_xticks(tick_positions)
    # Ensure features and x_ticks lists align in length
    ax.set_xticklabels(x_ticks[:len(tick_positions)], rotation=45, ha='right')
    ax.set_ylabel('Marginal Effect')
    ax.set_xlabel('')

    ax.set_xlim(min(tick_positions) - 0.75, max(tick_positions) + 0.75)

    # --- Custom Legend on Bottom ---
    handles = [plt.Line2D([0], [0], marker=party_markers[p], color=party_colors[p], linestyle='None') for p in parties]
    ax.legend(handles, parties, title='', loc='upper center', bbox_to_anchor=(0.5, 1.19),
              ncol=len(parties), frameon=False, markerscale=2.5)

    sns.despine()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        plot_filepath = os.path.join(output_dir, 'model_coefficients.pdf')
        plt.savefig(plot_filepath, bbox_inches='tight')
        print(f"model_coefficients plot saved to '{plot_filepath}'")
    plt.show()