import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from scipy.stats import wasserstein_distance
import matplotlib.patches as mpatches

def create_demographic_plot(df, parties, output_dir='imgs', save=False):
    """
    Generates and displays a set of demographic distribution plots for political parties.
    
    This function creates a grid of plots comparing the gender and age distribution of impressions
    for specified political parties against the complementary set of parties.
    """
    # Set the global theme for all plots in this function
    sns.set_theme(style="ticks", context="paper", font_scale=1.85)
    
    # --- Data Preprocessing ---
    df['demographic_distribution'] = df['demographic_distribution'].apply(ast.literal_eval)
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