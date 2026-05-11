# Classification Automatique de Pathologies Cardiaques - Challenge ACDC

Ce dépôt contient un pipeline complet d'apprentissage automatique dédié à la classification de pathologies cardiaques à partir d'examens d'IRM en ciné (cine-MRI). Ce travail s'appuie sur le jeu de données public du **Challenge ACDC (Automated Cardiac Diagnosis Challenge)** organisé lors de MICCAI 2017 : [https://www.creatis.insa-lyon.fr/Challenge/acdc/](https://www.creatis.insa-lyon.fr/Challenge/acdc/).

## Contexte Clinique et Scientifique

Le diagnostic de l'état cardiaque (fonction ventriculaire, cardiomyopathies, etc.) nécessite l'évaluation de paramètres physiologiques précis extraits de l'imagerie. Les algorithmes d'apprentissage automatique, bien que performants, ne peuvent être intégrés en pratique clinique sous forme de "boîtes noires". Un rapport médical doit toujours expliciter les raisons physiologiques justifiant un diagnostic. 

L'objectif de ce projet est double :
1. **Modélisation** : Développer un modèle de classification pour 5 classes cliniques (Sujet Sain [NOR], Infarctus du Myocarde [MINF], Cardiomyopathie Dilatée [DCM], Cardiomyopathie Hypertrophique [HCM], Anomalie du Ventricule Droit [RV]).
2. **Explicabilité** : Fournir une interprétabilité forte des prédictions en identifiant l'importance clinique des variables extraites (surfaces ventriculaires et myocardiques en fin de diastole et systole).

## Architecture du Projet

Le dépôt est structuré de la manière suivante pour garantir la séparation entre l'extraction lourde des données et l'analyse exploratoire :

```text
acdc-cardiac-classification/
├── data/                               # Dossier des données
│   └── dataset_complet_patients.csv    # Dataset consolidé (généré par main.py)
├── notebooks/                          
│   └── analyse_exploratoire_et_resultats.ipynb  # Analyse EDA, perfs et SHAP
├── src/                                # Modules métier
│   ├── data_processing.py              # Parsing des configurations et fusion des données
│   ├── extraction.py                   # Extraction radiomique (pyradiomics)
│   └── modeling.py                     # Modélisation (LogReg, CV) et SHAP
├── main.py                             # Script d'orchestration global
├── features.yaml                       # Configuration d'extraction pyradiomics
├── requirements.txt                    # Dépendances Python
└── README.md
```

(Note : Les données NIfTI brutes et les dossiers patients doivent être placés dans data/ mais ne sont pas suivis par Git pour des raisons de volumétrie et de bonnes pratiques de recherche).

## Installation et Reproductibilité

Clonez le dépôt et créez un environnement virtuel

git clone [https://github.com/votre-nom-utilisateur/acdc-cardiac-classification.git](https://github.com/votre-nom-utilisateur/acdc-cardiac-classification.git)
cd acdc-cardiac-classification
python -m venv venv
source venv/bin/activate  # Sous Windows : venv\Scripts\activate

## Attention - Installation de PyRadiomics :

### 1. Installer pyradiomics depuis la source
pip install git+[https://github.com/AIM-Harvard/pyradiomics.git](https://github.com/AIM-Harvard/pyradiomics.git)

### 2. Installer les autres dépendances
pip install -r requirements.txt

## Utilisation

Placez les dossiers patients du challenge ACDC (contenant les fichiers .nii et Info.cfg) directement dans le répertoire data/ (ex: data/patient001/).

Lancez le script d'orchestration :

python main.py

Le script exécutera séquentiellement :

    L'extraction itérative des surfaces radiomiques (Ventricule Droit, Myocarde, Ventricule Gauche).

    La consolidation des données avec les paramètres cliniques.

    La sauvegarde du fichier dataset_complet_patients.csv.

    L'évaluation du modèle par validation croisée stratifiée (100 itérations).

    Ouvrez ensuite le notebook notebooks/analyse_exploratoire_et_resultats.ipynb pour visualiser l'analyse des données, la stabilité du modèle et les graphes de l'interprétabilité SHAP.

## Résultats et Interprétabilité

Le modèle de référence choisi est une Régression Logistique fortement régularisée, adaptée aux petits jeux de données médicaux. Évalué sur 100 splits de validation croisée, le modèle obtient d'excellents scores AUC (Area Under the Curve) :
    
    HCM & RV : > 0.98
    
    DCM & NOR : > 0.96

    MINF : ~ 0.95

Explicabilité (SHAP) :
L'analyse démontre que l'algorithme s'appuie sur des heuristiques cliniquement valides :

    La surface du ventricule gauche en fin de systole (LV_ES) est le prédicteur principal pour la quasi-totalité des classes (déterminant majeur de la fraction d'éjection).

    La prédiction de la classe RV est pilotée spécifiquement par la géométrie du ventricule droit en fin de systole (RV_ES).

    Les caractéristiques macroscopiques (taille, poids) ont une influence marginale, confirmant la nécessité de l'imagerie ciblée.
