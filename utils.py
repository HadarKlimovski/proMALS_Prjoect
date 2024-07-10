

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




