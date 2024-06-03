

# ProMALS : Advanced Proteomic Profiling for Precision Biomarkers in ALS Prognosis and Disease Progression

### Abstract

Amyotrophic lateral sclerosis (ALS), is a progressive neurodegenerative disorder characterised by the death of motor neurons in the brain and spinal cord. This cell death leads to a gradual loss of muscle control, and usually results in death by respiratory failure within three to five year on average from symptoms onset.The Interest in biomarkers relevant to ALS has grown steadily over the past decade1. Proteomics methods based on mass spectrometry hold special promise for the discovery of novel biomarkers that might form the foundation for new clinical blood tests, but to date, they are disappointing. This is due in part to the lack of a coherent pipeline connecting marker discovery with well-established methods for validation.  
    Indeed, Better diagnostic markers for ALS are urgently needed to improve diagnosis, guide molecularly targeted therapy and monitor activity and therapeutic16, guide molecularly targeted therapy and monitor activity and therapeutic response across a wide spectrum of disease, in our case, ALS. ProMALS is developed to understand ALS through the discovery of precision protein biomarkers. This project utilizes mass spectrometry technology to identify and validate protein biomarkers from patients serum that can predict the progression of ALS and aid in prognosis, offering new possibilities for personalized therapeutic strategies.


### Aims

1.Developing candidate biomarkers from human liquid biopsies for ALS progression, based on unbiased proteomic profiling.

Hypothesis: Pathology relevant proteins can be found and analyzed in ALS patients sera as biomarkers for patients survival.

### Discovery 
I plan on exploring the relationship of proteins to patients survival with linear and non-linear models including Cox regression,and  Survival-XGBoost. Each model allows us to assess the importance of each protein in predicting a patient’s survival probability, from which we can extract those that are most relevant to our outcomes.

### Data procesing 
1.1. 2929 proteins were annotated based on human proteome and uniprot database across 179 ALS patients.

1.2. filertation : 
- Filer out samples with 10% - 50% missing values.
- Filter out immune system proteins that have hight variability and are not disease related  to prevent introducing noise to the model’s training [ IGKLV, IGKV, KERATIN, IGLV,IGHV).
1.3. Imputing missing Values(optionaly) Replace NaN values with a substitute value such as the mean, median, or mode of the column.
1.4. normalization by total protein abundance of the sample.
1.5. log2 transformation.


### Feature selection : Univariate Cox regresssion 
2.1.This approach allows systematic evaluation of each protein's impact on survival, helping to pinpoint candidates for deeper biological or clinical investigation.
- Select proteins based on their statistical significance and effect size (hazard ratio), focusing on those possible greater correlation to patients’ survival.
2.2. Alternatively, performing PCA and extracting most heterogeneous protein across our cohort following by analyses of protein levels and patients’ survival.

### Models 
3.1. Cox regression : linear model that calculates individual survival probability per patient and can estimate the hazard ratio of each protein, a statistic depicting the correlation between a protein’s level and patient’s survival.
3.2. Survival-XGBoost : EXtreme Gradient Boosting (XGBoost) aims to predict a target variable according to a set of features and their linear and non-linear connection.. Survival-XGBoost is a specialized variant of the XGBoost algorithm, adapted specifically for survival analysis. This adaptation enables it to handle right-censored data commonly encountered in survival scenarios.By incorporating our  data, Survival-XGBoost can make effective predictions about time-to-event outcomes.

### Outcome Analysis:
-	In my application of these three methods, I aim to identify key features that influence the progression of ALS disease. The predictors (features) include protein abundance as well as clinical features such as sex,patients progressive state as measured by  deltaFRS, and disease format (bulbar or limb). Survival time serves as the target variable.
-	I will perform SHAP analysis to explain the output of machine learning models by measuring the impact of each feature on the prediction. SHAP values can help understand how each feature influences the model's prediction of an event occurring.






