import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from configs import CLINICAL_DATA, IMMUNE_PROTEINS, KERATIN



    
    
def load_raw_proteomics_data():
    # Load
    print(f"Loading raw data: {"report_DIANN_perseus.xlsx"}")
    df_original = pd.read_excel("report_DIANN_perseus.xlsx")
    print(f"Raw data shape: {df_original.shape}")

    # Drop columns
    df_original.drop(columns=["Unnamed: 0"], inplace=True)

    # Split DataFrame into two parts
    df1 = df_original.iloc[:, :182]  # Assuming '182' is the correct end column index
    df2 = df_original.iloc[:, 186:]  # Assuming '186' is the correct start column index

    # Process df1 by dropping the first four rows and resetting index
    df1_processed = df1.iloc[4:].reset_index(drop=True)

    # Extract protein names and merge with processed df1
    protein_names_df = df2[['Genes']]
    protein_names_df = df2[['Genes']]
    protein_names_df = protein_names_df.iloc[4:].reset_index(drop=True)
    merged_df = pd.merge(df1_processed, protein_names_df, left_index=True, right_index=True, how='inner')

    # Further processing of the merged DataFrame
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.rename(columns={'Genes': 'sample_ID'}, inplace=True)
    merged_df = merged_df.set_index('sample_ID')
    merged_df = merged_df.transpose()  # Transpose the DataFrame
    merged_df.replace(13, np.nan, inplace=True)  # Replace '13' with NaN

    print(f"Processed data shape: {merged_df.shape}")
    return merged_df



def rename_columns(df):
    
    # Replace NaN column names with 'unidentified'
    df.columns = ['unidentified' if pd.isna(x) else x for x in df.columns]

    # Find duplicate column names and rename them
    duplicates = df.columns.duplicated(keep=False)
    seen = {}
    new_names = []
    for col, dup in zip(df.columns, duplicates):
        if dup:
            if col in seen:
                seen[col] += 1
                new_name = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
                new_name = col
            new_names.append(new_name)
        else:
            new_names.append(col)

    df.columns = new_names
    return df



def describe_proteins_and_samples(df):
   
    # Use describe() to calculate descriptive statistics for proteins (columns)
    # protein_stats = df.describe()
    # print(protein_stats)
    
    # Extract mean and variance for histogram plotting
    protein_means = df.mean()
    protein_variances = df.var()
    

    # Plot the distribution of protein means
    plt.figure(figsize=(8, 4))
    plt.hist(protein_means, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Protein Means')
    plt.xlabel('Mean Protein Abundance')
    plt.ylabel('Frequency')
    plt.show()

    # Plot the distribution of protein variances
    plt.figure(figsize=(8, 4))
    plt.hist(protein_variances, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Protein Variances')
    plt.xlabel('Variance of Protein Abundance')
    plt.ylabel('Frequency')
    plt.show()

    # Calculate statistics for samples (rows)
    sample_protein_counts = df.count(axis=1)

    # Plot the distribution of protein counts per sample
    plt.figure(figsize=(8, 4))
    plt.hist(sample_protein_counts, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Protein Counts per Sample')
    plt.xlabel('Number of Detected Proteins per Sample')
    plt.ylabel('Frequency of Samples')
    plt.show()
    
    

    # Plotting the box plot for protein means
    plt.figure(figsize=(10, 6))
    plt.boxplot(protein_means)
    plt.title("Box Plot of Protein Means")
    plt.ylabel("Protein Abundance")
    plt.grid(True)
    plt.show()

    # Collect and return the statistics
    stats = {
        'protein_descriptive_stats': df,
        'protein_means': protein_means,
        'protein_variances': protein_variances
    }

    return stats



def drop_protein(df, threshold=0.005):
    """
    Drops specified protein columns, filters out columns based on the count of non-NA values and variance threshold.
    Additionally sets 'sample number' as the DataFrame index and plots the variance distribution with the threshold.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    IMMUNE_PROTEINS (list): List of immune protein columns to drop.
    KERATIN (list): List of keratin columns to drop.
    threshold (float): Variance threshold for filtering columns.
    """
    # Drop immune protein columns and print the shape
    df = df.drop(columns=IMMUNE_PROTEINS)
    print(f"Shape after dropping immune proteins: {df.shape}")

    # Drop keratin columns and print the shape
    df = df.drop(columns=KERATIN)
    print(f"Shape after dropping keratin: {df.shape}")

    # Filter out proteins that are found in less than 50% of the tested samples
    required_count = len(df) * 0.5
    df = df.loc[:, df.count() >= required_count]
    print(f"Shape after filtering low prevalence proteins: {df.shape}")

    # Calculate variance and apply variance threshold filtering - Remove proteins with variance lower than threshold
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    df = df[df.columns[selector.get_support(indices=True)]]
    print(f"Proteins with variance above threshold (threshold={threshold}): {df.shape}")
    
    # Plot the distribution of variances with a threshold line
    variance = df.var()
    plt.figure(figsize=(12, 6))
    plt.hist(variance, bins=40, alpha=0.7, edgecolor='black')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    plt.title('Distribution of Protein Variances with Threshold Line')
    plt.xlabel('Variance of Protein Abundance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Reset index, rename 'index' to 'sample number', and set 'sample number' as the index
    df = df.reset_index().rename(columns={'index': 'sample number'})
    df = df.set_index("sample number")

    return df



def norm_and_log(df):
    # Normalize 1
    row_sums = df.abs().sum(axis=1)
    normalized_df = df.div(row_sums, axis=0)

    # Check that the sum of each row is 1
    row_sums_check = normalized_df.sum(axis=1)
    print(f"Sum of rows is 1: {np.allclose(row_sums_check, 1)}")
    print(f"Sum of the first 3 rows after normalization:{row_sums_check.head(3)}")
    # Apply log2 transformation, adding a small value to avoid log(0)
    df = np.log2(normalized_df + 1e-10)  # Adding a tiny value to avoid log2(0)
    
    return df


def fill_NA(df):
    
    return df.fillna(0.00001)


# def load_and_merge_with_clinical(proteomics_data):
#     clinical_data = pd.read_csv(CLINICAL_DATA)
#     print(f"Loading clinical data from {CLINICAL_DATA}\nShape: {clinical_data.shape}")
    
#     clinical_data = clinical_data[["sample number", "Survival_from_onset (months)","Status dead=1"]]
#     clinical_data.set_index("sample number", inplace=True)
#     df_with_clinical = proteomics_data.join(clinical_data, how='inner') 
#     return df_with_clinical

def describe_clinical_data(df):
     # Plotting the box plot for "Survival_from_onset (months)"
    plt.figure(figsize=(10, 6))
    plt.boxplot(df["Survival_from_onset (months)"])
    plt.title("Box Plot of Survival from Onset (months)")
    plt.ylabel("Survival from Onset (months)")
    plt.grid(True)
    
    plt.figure(figsize=(8, 4))
    plt.hist(df["Survival_from_onset (months)"], bins=10, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Survival time from Onset')
    plt.xlabel('Survival time(month)')
    plt.ylabel('Frequency of Samples')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    df.boxplot(column="Survival_from_onset (months)", by="Sex", grid=False)
    num_males = df[df['Sex'] == 0].shape[0]
    num_females = df[df['Sex'] == 1].shape[0]
    plt.text(1, df["Survival_from_onset (months)"].max(), f'N = {num_males}', horizontalalignment='center', verticalalignment='center')
    plt.text(2, df["Survival_from_onset (months)"].max(), f'N = {num_females}', horizontalalignment='center', verticalalignment='center')
    
    plt.title("Box Plot of Survival from Onset (months) by Sex")
    plt.suptitle("")
    plt.xlabel("Sex")
    plt.ylabel("Survival from Onset (months)")
    plt.xticks([1, 2], ["Male", "Female"])  # Set custom x-tick labels
    plt.grid(True)
    plt.show()
    
    # Histogram for "Age Onset (years)" with average line
    plt.figure(figsize=(8, 4))
    plt.hist(df["Age Onset (years)"], bins=10, edgecolor='black', alpha=0.7)
    average_age_onset = df["Age Onset (years)"].mean()
    plt.axvline(average_age_onset, color='red', linestyle='dotted', linewidth=2)
    #plt.text(average_age_onset, plt.ylim()[1] * 0.9, f'Average: {average_age_onset:.2f} years', color='red', horizontalalignment='right')
    plt.title('Distribution of Age Onset (years)')
    plt.xlabel('Age Onset (years)')
    plt.ylabel('Frequency of Samples')
    plt.show()
    
    return df

def prepare_data_to_model(proteomics_data):
    clinical_data = pd.read_csv(CLINICAL_DATA)
    print(f"Loading clinical data from {CLINICAL_DATA}\nShape: {clinical_data.shape}")
    
    clinical_data = clinical_data[["sample number", "Survival_from_onset (months)","Status dead=1", "Sex", "Disease Format", "ALSFRS score (unit)", "ALSFRS_group","Age Onset (years)"]]
    clinical_data.set_index("sample number", inplace=True)
    df_with_clinical = proteomics_data.join(clinical_data, how='inner')
    sex_mapping = {'M': 0, 'F': 1}
    df_with_clinical['Sex'] = df_with_clinical['Sex'].map(sex_mapping)
    disease_format_mapping = {'Limb': 0, 'Bulbar': 1}
    df_with_clinical['Disease Format'] = df_with_clinical['Disease Format'].map(disease_format_mapping)

    return df_with_clinical

def standardize(df) :
    columns_to_exclude = ['Status dead=1', 'Sex']
    # Separate the columns to exclude from standardization




