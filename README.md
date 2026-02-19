# The direct democracy paradox: Microtargeting and issue ownership in Swiss online political ads

This repository contains the source code and analysis scripts for the paper:  
**The direct democracy paradox: Microtargeting and issue ownership in Swiss online political ads.**

## 📊 Project Overview
This study presents a large-scale, data-driven analysis of 40k political ads published on Facebook and Instagram in Switzerland between 2021 and 2025. We investigate:
* **Microtargeting (RQ1):** Partisan strategies stratified by demographic (age, gender) and linguistic region.
* **Referenda (RQ2):** The impact of ad intensity on direct democratic outcomes.
* **Issue Ownership (RQ3):** Patterns of topic divergence and the predictability of ad authorship using machine learning.

## 📂 Repository Structure
* The repository is organized as follows:

```
repo/
├── src/
│   ├── __init__.py
│   ├── models.py          # Statistical models and analysis functions
│   ├── plot_functions.py  # Visualization functions
│   └── prepare_data.py    # Data loading and cleaning logic
├── process_data.py        # Main script to process data - entry point
├── RQ1a_RQ1b.ipynb        # Analysis notebook for RQ1
├── RQ2a_RQ2b.ipynb        # Analysis notebook for RQ2
├── RQ3a.ipynb             # Analysis notebook for RQ3a
├── RQ3b.ipynb             # Analysis notebook for RQ3b
└── data/                  # Data folder
```

## 🚀 Usage
Before running the notebooks, process the data by running:
```bash
python process_data.py
```

## Data Access
To comply with the FAIR principles of persistent and citable data, the datasets underlying this analysis are archived on **Zenodo**.

**Data Link:** 10.5281/zenodo.18702491

### Data Files
The Zenodo repository includes the following files:

* **ads.zip**: A compressed archive containing the raw text and metadata of the collected advertisements. The data is organized into three sub-folders by language: German (DE), French (FR), and Italian (IT).

* **federal_elections_authors_annotation.csv**: A manually curated list of authors who published at least one ad in the 30 days preceding the 2023 Swiss federal elections. Each author is labeled with:
  * Political affiliation.
  * Relevance to the federal election.
  * Entity type (Party, NGO, Newspaper, Committee, Union, or Association).
  * Specific linkage to a political party or individual politician.

* **federal_gpt_answ.json**: Results of an automated annotation process using GPT-4o for ads published within the 30-day window of the federal election. It identifies whether an ad is election-relevant and provides up to three thematic keywords describing the ad's topic.

* **referendum_gpt_answ.json**: Annotated data for ads published within 30 days of the 42 referenda held during the study period. This file identifies the ad's relevance to a specific referendum and its stance ("Yes", "No", or "Neutral") as determined by GPT-4o.

* **referendums_topic.json**: Official dataset containing the title, date, and description for each referendum.

* **refimp_with_results_and_participation.csv**: Official dataset detailing the results and voter turnout for each referendum.

* **topic_mapping.json**: A classification of referendums into 12 primary categories, clustered using GPT-4o.


## 📜 Citation
If you use this code or data, please cite:
> Capozzi, A. (2025). The direct democracy paradox: Microtargeting and issue ownership in Swiss online political ads. arXiv preprint arXiv:2512.14564.
> 
> @article{capozzi2025talking,
>
>   title={The direct democracy paradox: Microtargeting and issue ownership in Swiss online political ads},
>
>   author={Capozzi, Arthur},
>
>   journal={arXiv preprint arXiv:2512.14564},
> 
>   year={2025}
> 
> }
## ⚖️ License
The code in this repository is licensed under the **MIT License**.
