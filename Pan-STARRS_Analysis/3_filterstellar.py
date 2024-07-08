import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from astropy.coordinates import SkyCoord
import astropy.units as u

def remove_non_stellar_types(df):
    stellar_types = ['Star', 'RGB*', 'Variable*', 'HighPM*', 'ChemPec*', 'HorBranch*', 'Low-Mass*',
                     'HorBranch*_Candidate', 'EclBin_Candidate', 'EclBin', 'PulsV*', 'BlueStraggler',
                     'RSCVnV*', 'WhiteDwarf', 'RRLyrae', 'BrownD*_Candidate', 'delSctV*', 'HotSubdwarf',
                     'Type2Cep', 'BYDraV*', 'HighVel*', 'LongPeriodV*', 'SB*', 'HotSubdwarf_Candidate',
                     'C*', 'C*_Candidate', 'WhiteDwarf_Candidate', 'LongPeriodV*_Candidate', 'AGB*',
                     'EllipVar', 'Cepheid', 'MainSequence*', 'alf2CVnV*', 'RGB*_Candidate', 'SXPheV*',
                     'post-AGB*', 'RRLyrae_Candidate', 'Mira', 'RotV*']
    return df[df['object_type'].isin(stellar_types)]

def plot_cmd_with_labels(label_stellar_df):
    fig = plt.figure(figsize=(10, 8))
    
    plt.scatter(label_stellar_df['gMeanPSFMag'] - label_stellar_df['iMeanPSFMag'], label_stellar_df['gMeanPSFMag'], s=1, alpha=0.1, label='All Data')
    
    labels_to_show = ['HorBranch*', 'RGB*', 'Low-Mass*', 'BlueStraggler', 'HotSubdwarf', 'RRLyrae']
    
    for obj_type in labels_to_show:
        if obj_type in label_stellar_df['object_type'].unique():
            obj_type_data = label_stellar_df[label_stellar_df['object_type'] == obj_type]
            plt.scatter(obj_type_data['gMeanPSFMag'] - obj_type_data['iMeanPSFMag'], obj_type_data['gMeanPSFMag'], s=10, alpha=0.8, label=obj_type)
    
    plt.xlabel('g - i')
    plt.ylabel('g')
    plt.title('g vs g-i (SIMBAD Labels)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1), ncol=2)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'cmd_with_labels.png'), dpi=300)
    plt.close()

def find_non_stellar_types(df):
    stellar_types = ['Star', 'RGB*', 'Variable*', 'HighPM*', 'ChemPec*', 'HorBranch*', 'Low-Mass*',
                     'HorBranch*_Candidate', 'EclBin_Candidate', 'EclBin', 'PulsV*', 'BlueStraggler',
                     'RSCVnV*', 'WhiteDwarf', 'RRLyrae', 'BrownD*_Candidate', 'delSctV*', 'HotSubdwarf',
                     'Type2Cep', 'BYDraV*', 'HighVel*', 'LongPeriodV*', 'SB*', 'HotSubdwarf_Candidate',
                     'C*', 'C*_Candidate', 'WhiteDwarf_Candidate', 'LongPeriodV*_Candidate', 'AGB*',
                     'EllipVar', 'Cepheid', 'MainSequence*', 'alf2CVnV*', 'RGB*_Candidate', 'SXPheV*',
                     'post-AGB*', 'RRLyrae_Candidate', 'Mira', 'RotV*']
    
    df = df[~df['object_type'].isin(stellar_types)]
    print(f"Found {len(df)} non-stellar objects.")
    return df

if __name__ == '__main__':

    DATA_DIR = "/N/project/catypGC/Bradley/Data_ML"
    PLOT_DIR = "/N/project/catypGC/Bradley/Plots_ML"

    try:
        PS_clean = pd.read_csv(os.path.join(DATA_DIR, 'PS_clean.csv'))
        simbad = pd.read_csv(os.path.join(DATA_DIR, 'simbad_matched.csv'))
    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
        exit(1)
    except pd.errors.EmptyDataError:
        print("Error: One of the CSV files is empty.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the files: {e}")
        exit(1)

    bands = ['g', 'r', 'i', 'z', 'y']
    
    simbad_combined = pd.merge(PS_clean, simbad, left_on='objID', right_on='PS_id', how='inner')
    # Print all unique object types
    print(simbad_combined['object_type'].unique())
    
    simbad_combined = remove_non_stellar_types(simbad_combined)
    print(f"Combined dataframe with SIMBAD data contains {len(simbad_combined)} entries.")
    
    try:
        simbad_combined.to_csv(os.path.join(DATA_DIR, 'simbad_cleaned.csv'), index=False)
    except Exception as e:
        print(f"Error saving simbad_cleaned.csv: {e}")

    non_stellar_df = find_non_stellar_types(simbad_combined)
    non_stellar_objIDs = non_stellar_df['objID'].unique()
   
    PS_clean_stellar = PS_clean[~PS_clean['objID'].isin(non_stellar_objIDs)]
    print(f"Merged dataframe with non-stellar objects removed contains {len(PS_clean_stellar)} entries.")
    
    label_stellar_df = simbad_combined[simbad_combined['objID'].isin(PS_clean_stellar['objID'])]
    print(f"Filtered dataframe with only stellar types contains {len(label_stellar_df)} entries.")
    plot_cmd_with_labels(label_stellar_df)
    
    try:
        PS_clean_stellar.to_csv(os.path.join(DATA_DIR, 'PS_clean_NN.csv'), index=False)
        simbad_combined[['objID', 'object_type']].to_csv(os.path.join(DATA_DIR, 'simbad_filtered.csv'), index=False)
    except Exception as e:
        print(f"Error saving output files: {e}")