import pandas as pd
import os

def load_and_clean_data():
    data_dir = 'data'
    
    print("1. Loading datasets...")
    # loading the mapping file that connects ModelID to CosmicID (DepMap IDs)
    model_df = pd.read_csv(os.path.join(data_dir, 'Model.csv'))[['ModelID', 'COSMICID']]
    
    # loading the drug response file (GDSC2)
    drug_df = pd.read_csv(os.path.join(data_dir, 'GDSC2_fitted_dose_response_27Oct23.csv'))
    
    print("2. Mapping Drug Data to DepMap IDs...")
    #mapping the drug response data to the model data using CosmicID as the key
    model_df = model_df.rename(columns={'COSMICID': 'COSMIC_ID'})
    
    #join to keep only drug response records that have a corresponding model in the model_df
    drug_with_models = pd.merge(drug_df, model_df, on='COSMIC_ID', how='inner')
    print(f"   Mapped {len(drug_with_models)} drug records to DepMap IDs.")

    # loading the gene expression data (TPM log-transformed) for human protein-coding genes
    print("3. Loading Expression Data (this might take a moment)...")
    #loading the gene expression data, which is expected to have ModelID as the index and gene names as columns
    expr_df = pd.read_csv(os.path.join(data_dir, 'OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv'))
    expr_df = expr_df.rename(columns={expr_df.columns[0]: 'ModelID'})
    
    # renaming the index column to 'ModelID' for easier merging later on
    #expr_df = expr_df.rename(columns={'Unnamed: 0': 'ModelID'})
    
    print("4. Final Merge: Joining Drug response with Expression...")
    # merging the drug response data with the gene expression data on 'ModelID' to create a final dataset for modeling
    print("3. Final Merge: Drug Response + Gene Expression...")
    # This merge will keep only the records that have both drug response data and gene expression data, ensuring we have a complete dataset for downstream analysis.
    final_df = pd.merge(drug_with_models, expr_df, on='ModelID', how='inner')
    
    print(f"--- Process Complete ---")
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Unique Drugs: {final_df['DRUG_NAME'].nunique()}")
    print(f"Unique Cell Lines: {final_df['ModelID'].nunique()}")
    
    # Saving the final processed dataset to a new CSV file for use in modeling
    output_path = os.path.join(data_dir, 'processed_summary.csv')
    final_df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    load_and_clean_data()