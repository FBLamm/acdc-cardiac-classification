import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

def extract_surfaces(data_dir: str, yaml_params: str) -> pd.DataFrame:
    """Extrait la surface des régions d'intérêt pour chaque patient et frame."""
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    extractor = featureextractor.RadiomicsFeatureExtractor(yaml_params)
    labels = [1, 2, 3] # 1: RV cavity, 2: myocardium, 3: LV cavity
    results = []

    for patient_folder in sorted(os.listdir(data_dir)):
        patient_path = os.path.join(data_dir, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        patient_id = patient_folder

        # Identification des frames nifti (hors ground truth)
        frames = [f.split("_frame")[1].split(".")[0] for f in os.listdir(patient_path)
                  if f.startswith(f"{patient_id}_frame") and f.endswith(".nii") and not f.endswith("_gt.nii")]

        for frame in sorted(frames):
            img_path = os.path.join(patient_path, f'{patient_id}_frame{frame}.nii')
            gt_path = os.path.join(patient_path, f'{patient_id}_frame{frame}_gt.nii')

            if os.path.exists(gt_path):
                try:
                    img = sitk.ReadImage(img_path)
                    img_gt = sitk.ReadImage(gt_path)

                    for lbl in labels:
                        try:
                            features = extractor.execute(img, img_gt, label=lbl)
                            surface_area = features.get('original_shape_SurfaceArea')
                            if surface_area is not None:
                                results.append({
                                    'Patient_ID': patient_id,
                                    'Frame': frame,
                                    'Label_ID': lbl,
                                    'SurfaceArea': surface_area.item()
                                })
                        except Exception:
                            pass
                except Exception as e:
                    print(f"Erreur de lecture pour {patient_id} (Frame {frame}): {e}")

    df_surface = pd.DataFrame(results)

    # Pivot pour obtenir les surfaces par label en colonnes
    df_wide = df_surface.pivot(index=['Patient_ID', 'Frame'], columns='Label_ID', values='SurfaceArea')
    df_wide.columns = [f'SurfaceArea_{col}' for col in df_wide.columns]

    return df_wide.reset_index()
