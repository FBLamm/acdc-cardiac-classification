import os
from sklearn.model_selection import train_test_split
from src.data_processing import parse_info_files, build_final_dataset
from src.extraction import extract_surfaces
from src.modeling import evaluate_model_cv, plot_shap_summary

def main():
    data_dir = 'data'
    yaml_params = 'features.yaml'
    output_csv = os.path.join(data_dir, 'dataset_complet_patients.csv')

    print("1. Parsing des métadonnées patients...")
    info_df = parse_info_files(data_dir)

    print("2. Extraction des features radiomiques (cela peut prendre un moment)...")
    features_df = extract_surfaces(data_dir, yaml_params)

    print("3. Construction du dataset final...")
    df_final = build_final_dataset(info_df, features_df)

    print(f"Sauvegarde du dataset dans : {output_csv}")
    df_final.to_csv(output_csv, index=False)

    # Sélection des features
    feature_cols = [
        'Height', 'Weight',
        'SurfaceArea_1_ED', 'SurfaceArea_1_ES',
        'SurfaceArea_2_ED', 'SurfaceArea_2_ES',
        'SurfaceArea_3_ED', 'SurfaceArea_3_ES'
    ]
    X = df_final[feature_cols]
    y = df_final['Group']

    print("4. Évaluation du modèle par validation croisée (100 splits)...")
    auc_scores = evaluate_model_cv(X, y, n_splits=100)
    for cls, scores in auc_scores.items():
        print(f"Classe {cls} - AUC moyen : {sum(scores)/len(scores):.3f}")

    print("5. Analyse de l'interprétabilité avec SHAP...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    plot_shap_summary(X_train, X_test, y_train)

if __name__ == "__main__":
    main()
