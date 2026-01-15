# "Talking Past Each Other": Issue Ownership and Microtargeting in Swiss Online Political Ads

This repository contains the source code and analysis scripts for the paper:  
**"Talking Past Each Other": Issue Ownership and Microtargeting in Swiss Online Political Ads.**

## 📊 Project Overview
This study presents a large-scale, data-driven analysis of 40k political ads published on Facebook and Instagram in Switzerland between 2021 and 2025. We investigate:
* **Microtargeting (RQ1):** Partisan strategies stratified by demographic (age, gender) and linguistic region.
* **Referenda (RQ2):** The impact of ad intensity on direct democratic outcomes.
* **Issue Ownership (RQ3):** Patterns of topic divergence and the predictability of ad authorship using machine learning.

## 📂 Repository Structure
* WIP

## 🛠️ Data Access
To comply with the FAIR principles of persistent and citable data, the datasets underlying this analysis are archived on **Zenodo**.

| File | Description |
| :--- | :--- |
| `ads.zip` | Raw text and metadata of ads disaggregated by language (German, French, Italian). |
| `federal_elections_authors_annotation.csv` | Manual labels for 943 pages (party affiliation, entity type, etc.). |
| `federal_election_gpt_topic.json` | GPT-4o annotations for election relevance and topic keywords. |
| `referendum_gpt_stance.json` | GPT-4o annotations for referendum relevance and Yes/No stance. |

**Data Link:** 10.5281/zenodo.18256330


## 📜 Citation
If you use this code or data, please cite:
> Capozzi, A. (2025). "Talking past each other": Issue ownership and microtargeting in Swiss online political ads. arXiv preprint arXiv:2512.14564.
> @article{capozzi2025talking,
>   title={" Talking past each other": Issue ownership and microtargeting in Swiss online political ads},
>   author={Capozzi, Arthur},
>   journal={arXiv preprint arXiv:2512.14564},
>   year={2025}
> }
## ⚖️ License
The code in this repository is licensed under the **MIT License**.
