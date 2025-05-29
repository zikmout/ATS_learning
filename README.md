ATOS ML Interview Playbook

A hands‑on, interview‑focused collection of Jupyter Notebooks


📑 About

This repository gathers concise, runnable notebooks that walk you through the machine‑learning algorithms most likely to surface during an ATOS (or any consulting/tech firm) data‑science interview. Each notebook follows the same pattern:

Implementation — scikit‑learn pipeline

Visual diagnostics — decision boundaries, feature importance, error curves


🗂 Notebook line‑up

#

Topic

File

What you’ll learn

01

Random Forest

01_random_forest.ipynb

Feature importance, OOB error, overfitting signals

02

Support Vector Machine (linear/RBF)

02_svm.ipynb

Kernel trick intuition, margin vs. C, grid‑search demo

03

Anomaly Detection (Isolation Forest)

03_isolation_forest.ipynb

Score interpretation, contamination ratio, comparison with One‑Class SVM

04

k‑Means Clustering

04_kmeans.ipynb

Elbow vs. silhouette, initialization pitfalls, cluster diagnostics

05

Feature Engineering

05_feature_eng.ipynb

ColumnTransformer, target encoding, pipeline tips

06

Model Validation

06_cross_val.ipynb

Stratified K‑fold vs. time‑series split, custom scorers

Tip: open any notebook in JupyterLab and run it top‑to‑bottom; no external datasets needed—the examples download or generate what they need.

🚀 Quick Start

# clone the repo
$ git clone https://github.com/zikmout/atos‑ml‑playbook.git
$ cd atos‑ml‑playbook

# set up the environment (Python >=3.10)
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt

# launch JupyterLab
$ jupyter lab

Open a notebook, hit Run All ( ▶️ ), and follow along with the inline commentary.