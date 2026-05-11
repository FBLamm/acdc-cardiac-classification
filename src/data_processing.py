import os
import pandas as pd
import numpy as np

def parse_info_files(data_dir: str) -> pd.DataFrame:
    """Parcourt les dossiers patients pour extraire les métadonnées de Info.cfg."""
    data = []

    for root, _, files in os.walk(data_dir):
        if 'Info.cfg' in files:
            patient_id = os.path.basename(root)
            file_path = os.path.join(root, 'Info.cfg')

            patient_data = {'Patient_ID': patient_id}
            with open(file_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(': ', 1)
                        patient_data[key] = value
            data.append(patient_data)

    df = pd.DataFrame(data)
    # Conversion des types si nécessaire
    df['Height'] = pd.to_numeric(df['Height'])
    df['Weight'] = pd.to_numeric(df['Weight'])
    return df

def build_final_dataset(info_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les métadonnées et les features radiomiques, et pivote les phases ED/ES."""
    # Fusion des deux sources
    df_merged = pd.merge(features_df, info_df, on='Patient_ID', how='left')

    # Sécurité sur les types : on convertit tout en entier pour éviter les bugs
    # de comparaison entre '01' (string) et 1 (int)
    df_merged['Frame_int'] = df_merged['Frame'].astype(int)
    df_merged['ED_int'] = df_merged['ED'].astype(int)
    df_merged['ES_int'] = df_merged['ES'].astype(int)

    # 1. FILTRAGE VITAL : On ne garde QUE les frames de fin de diastole et systole
    df_filtered = df_merged[
        (df_merged['Frame_int'] == df_merged['ED_int']) |
        (df_merged['Frame_int'] == df_merged['ES_int'])
    ].copy()

    # 2. Identification de la phase
    df_filtered['Phase'] = np.where(df_filtered['Frame_int'] == df_filtered['ED_int'], 'ED', 'ES')

    # Colonnes à conserver fixes lors du pivot
    fixed_cols = ['Patient_ID', 'ED', 'ES', 'Group', 'Height', 'NbFrame', 'Weight']

    # 3. Pivot pour avoir une seule ligne par patient
    df_pivot = df_filtered.pivot(
        index=fixed_cols,
        columns='Phase',
        values=['SurfaceArea_1', 'SurfaceArea_2', 'SurfaceArea_3']
    )

    # Aplatissement des colonnes hiérarchiques (ex: SurfaceArea_1_ED)
    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]

    # On nettoie notre dataframe avant de le renvoyer
    df_final = df_pivot.reset_index()

    return df_final
