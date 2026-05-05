# MHCD: Mental Health Comorbidity Discovery

This repository contains the code for MHCD (Mental Health Comorbidity Discovery), an intelligent decision support system for quantitative cross-condition mental health analysis using Disorder-Aware Structured Dictionary Learning (DASDL) and Robust PCA fusion.

## 📄 Paper

Title: MHCD: A Geometry-Driven Intelligent System for Interpretable Mental Health Comorbidity Discovery

Status: Submitted to Knowledge-Based Systems

Authors: Muhammad Usman Khalid, Shariq Bashir, Mudasir Ahmad Wani, Kashish Ara Shakil, Seyedali Mirjalili

## 📊 Dataset

This project uses the Reddit Mental Health Dataset by Low et al. (2020).

Download: https://zenodo.org/records/3941387

Citation:

Low, D. M., Rumker, L., Talkar, T., Torous, J., Cecchi, G., & Ghosh, S. S. (2020). 
Natural Language Processing Reveals Vulnerable Mental Health Support Groups and 
Heightened Health Anxiety on Reddit During COVID-19: Observational Study. 
Journal of Medical Internet Research, 22(10), e22635.

## 🚀 Getting Started

Prerequisites

    MATLAB R2020a or later (R2024a recommended)
    Python 3.8+ (for sentence embeddings)
    sentence-transformers library (pip install sentence-transformers)
    Required MATLAB toolboxes:
        Statistics and Machine Learning Toolbox
        Signal Processing Toolbox

Installation

    Clone this repository:

git clone https://github.com/usmankhalid06/MHCD.git
cd MHCD

    Download the dataset from Zenodo

    Extract the dataset to your working directory

    Install Python dependencies:

pip install sentence-transformers pandas numpy

⚠️ Note: Pre-computed embedding files exceed GitHub's 25 MB upload limit and are not included in this repository. Users must generate the six per-model embeddings locally using get_sentence_embeddings.m before running the main analysis.

## 📁 File Structure

```
MHCD/
├── DASDL.m                       # Disorder-Aware Structured Dictionary Learning (proposed)
├── my_ACSD.m                     # Adaptive Consistent Sequential Dictionary Learning
├── my_ODL.m                      # Online Dictionary Learning (requires SPAMS)
├── SDL.m                         # Sparse Orthogonal Component Analysis (SOCA)
├── gICA_exp.m                    # Group ICA baseline
├── kmeans_clustering.m           # K-means clustering baseline
├── litekmeans_k.m                # Lightweight K-means utility
├── LSICA.m                       # ICA support routines
├── RobustPCA.m                   # Robust PCA fusion across embedding models
├── clean_reddit_post.m           # Text preprocessing
├── get_sentence_embeddings.m     # Sentence transformer interface
├── script_full_code_midCovid.m   # Full pipeline (with embedding generation)
├── script_half_code_midCovid.m   # Pipeline (assuming embeddings saved)
└── README.md
```

## 💻 Usage

Step 1: Preprocess Data

% Clean and preprocess Reddit posts
cleaned_text = clean_reddit_post(raw_posts);

Step 2: Generate Sentence Embeddings

% Generate embeddings for all six sentence transformer architectures
% (MiniLM-L6, MiniLM-L12, MPNet, BGE, E5, GTE)
embeddings = get_sentence_embeddings(cleaned_text);

Step 3: Run Main Analysis

% Execute complete pipeline
script_full_code_midCovid

This will:

    Load embeddings from the six per-model .mat files
    Estimate intrinsic dimensionality (TwoNN) per model
    Derive K_m and rho_m from the Carathéodory bound
    Run DASDL and all five baselines (group ICA, K-means, SOCA, ODL, ACSD)
    Apply Robust PCA fusion across the six per-model DASDL outputs
    Generate comorbidity matrices, contrast ratios, and Π-based atom interpretability outputs

## 📖 Core Functions

Proposed Method

    DASDL.m - Disorder-Aware Structured Dictionary Learning (three-way feedback between atoms D, codes X, and disorder-atom assignment Π)
    RobustPCA.m - Robust PCA fusion across six per-model comorbidity matrices

Baseline BSS Methods

    my_ACSD.m - Adaptive Consistent Sequential Dictionary Learning
    my_ODL.m - Online Dictionary Learning (requires SPAMS)
    SDL.m - Sparse Orthogonal Component Analysis (SOCA)
    gICA_exp.m - Group Independent Component Analysis
    kmeans_clustering.m - Divide-and-conquer K-means clustering

For ODL you need to download SPAMS toolbox from here https://thoth.inrialpes.fr/people/mairal/spams/ to run mexOMP and mexLasso. DASDL and the other baselines do not require SPAMS.

Utilities

    clean_reddit_post.m - Text preprocessing (remove HTML, URLs, formatting)
    get_sentence_embeddings.m - Generate sentence transformer embeddings
    litekmeans_k.m, LSICA.m - Supporting routines for clustering and ICA baselines

## 🔧 Key Parameters

    Dictionary size (K_m): C * floor(d_I_hat) / 3 from the Carathéodory bound
    Sparsity (rho_m): floor(d_I_hat) + 1 from the Carathéodory bound
    Π-threshold coupling (lambda_1): 0.7
    Π softmax temperature (lambda_2): 0.7
    Iterations: 30 for all dictionary learning methods
    Convergence: relative dictionary change < 0.05

## 📊 Outputs

The analysis generates:

    Per-model 9 × 9 comorbidity matrices for each BSS method
    The fused DASDL+RPCA consensus comorbidity matrix
    Disorder-atom soft assignment matrix Π for DASDL
    Cross-model agreement matrices (Pearson r)
    Contrast ratio (Q75/Q25) per method
    TwoNN intrinsic dimensionality estimates per disorder per model

## 🧪 Reproducing Results

To reproduce paper results:

% Ensure dataset is in path
addpath('path/to/reddit/data');

% Run main script
script_full_code_midCovid

% Results will be saved in figures/ directory

To reproduce the other three temporal datasets (seasonal-2018, seasonal-2019, pre-Covid), generate embeddings for the relevant temporal window using get_sentence_embeddings.m, save each as a .mat file following the naming convention used in the mid-Covid scripts, and adapt the main script accordingly.

## 📧 Contact

For questions or issues, please contact:

    Muhammad Usman Khalid: [mukhalid@imamu.edu.sa]

## 🙏 Acknowledgments

This work was supported and funded by the Deanship of Scientific Research at Imam Mohammad Ibn Saud Islamic University (IMSIU).

## 📚 Citation

If you use this code, please cite:

@article{khalid2025mhcd,
  title={MHCD: A Geometry-Driven Intelligent System for Interpretable Mental Health Comorbidity Discovery},
  author={Khalid, Muhammad Usman and Bashir, Shariq and Wani, Mudasir Ahmad and Shakil, Kashish Ara and Mirjalili, Seyedali},
  journal={Knowledge-Based Systems},
  year={2025},
  note={Submitted}
}
