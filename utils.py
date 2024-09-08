def prepare_data_to_model_rep(df):
    clinical_data = pd.read_excel("clinical_data_rep.xlsx")
    print(f"Shape: {clinical_data.shape}")
    clinical_data = clinical_data[["sample number", "Survival_from_onset (months)","Status dead=1", "Sex", "Disease Format", "ALSFRS score (unit)","Age Onset (years)"]]
    clinical_data.set_index("sample number", inplace=True)
    df_with_clinical = df.join(clinical_data, how='inner')
    sex_mapping = {'M': 0, 'F': 1}
    df_with_clinical['Sex'] = df_with_clinical['Sex'].map(sex_mapping)
    disease_format_mapping = {'Limb': 0, 'Bulbar': 1}
    df_with_clinical['Disease Format'] = df_with_clinical['Disease Format'].map(disease_format_mapping)
   
   
   
    return df_with_clinical
