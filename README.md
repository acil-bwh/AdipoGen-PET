🧬 # AdipoGen-PET
Quantification of PET Activation in Adipose Tissue from Non-Contrast CT Scans

AdipoGen-PET provides the official code accompanying our study on generating PET-like metabolic activation maps of adipose tissue directly from standard non-contrast CT scans. This repository implements a conditional Generative Adversarial Network (cGAN) designed to estimate regional metabolic activity—traditionally measured using ^18F-FDG PET—without requiring radiotracers, additional radiation exposure, or PET imaging infrastructure.

The workflow combines CT-based adipose segmentation, custom fat-focused loss functions, and paired PET/CT training to enable accurate prediction of standardized uptake values (SUV) within adipose depots. Trained and validated on two independent cohorts, the model yields activation maps that show strong agreement with ground-truth PET measurements across diverse adipose regions.

This framework provides a scalable and radiation-sparing alternative for evaluating brown adipose tissue (BAT) and white adipose tissue metabolic activity in both clinical and research settings. It supports population-based studies of BAT prevalence, cardiometabolic health, and disease progression using routine CT scans, enabling metabolic phenotyping without the imaging burden of PET.

🔍 Key Features
* cGAN-based PET prediction from non-contrast CT
* Adipose tissue segmentation with voxel-wise and regional activation estimation
* Fat-enhanced loss function tailored to metabolic signal prediction
* Cross-cohort validation demonstrating reproducibility and generalizability
* PET-like SUV maps without radiotracers or radiation beyond standard CT
* Tools for visualization, region-level quantification, and statistical evaluation
