"""
Segmentation des phases de vol (HMM 5-états, 2-Features)
Objectif : Utiliser RoC + Groundspeed pour un HMM robuste avec gestion des états au sol et paliers.
"""
import pandas as pd
import numpy as np
import pyarrow 
from pathlib import Path
from hmmlearn import hmm
import warnings
import argparse
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION (5 PHASES)
# ============================================
# Noms des 5 états : Gère Taxi/Sol et Palier (Level Flight)
PHASE_ORDER = ['Ground_Taxi', 'Climb', 'Cruise', 'Level_Flight', 'Descent_Approach'] 
N_STATES_FINAL = 5 # Définition du nombre d'états


# ============================================
# PRÉPARATION DES DONNÉES (AUCUN CHANGEMENT NÉCESSAIRE)
# ============================================

def prepare_flight_data_3phase_2feature(trajectory_df):
    """
    Prépare les données pour le HMM (quel que soit N_STATES). 
    Fait confiance aux données déjà lissées par FDA.
    """
    df = trajectory_df.copy()
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Gérer les NA/Inf qui auraient pu subsister (remplacement par 0 pour le calcul numpy)
    df['vertical_rate'] = df['vertical_rate'].fillna(0).replace([np.inf, -np.inf], 0)
    df['groundspeed'] = df['groundspeed'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Renommage pour les features du modèle
    df['ROC'] = df['vertical_rate']
    df['GS'] = df['groundspeed']

    # Sélection des features pour le HMM
    return df


# ============================================
# MODÈLE HMM (5 ÉTATS) - AVEC INITIALISATION PHYSIQUE
# ============================================

class HMM_5State_Robust:
    """
    HMM à 5 états (Ground_Taxi, Climb, Cruise, Level_Flight, Descent_Approach)
    """
    def __init__(self, n_states=N_STATES_FINAL):
        self.n_states = n_states
        self.model = None
        self.phase_names = PHASE_ORDER

    def _create_initial_probabilities(self):
        """
        Initialise la probabilité de départ (doit commencer par Ground/Taxi)
        [0] = Ground/Taxi
        """
        pi = np.zeros(self.n_states)
        pi[0] = 0.95 # Forte probabilité de commencer au sol (Taxi/Ground)
        pi[1] = 0.05 # Faible probabilité de commencer en montée
        return pi

    def _create_constrained_transition_matrix(self):
        """Matrice de transition contrainte pour 5 phases (modèle d'inertie)"""
        trans = np.zeros((5, 5))
        # 0=Ground, 1=Climb, 2=Cruise, 3=Level_Flight, 4=Descent_Approach
        
        # Ground (0) -> Ground (0) ou Climb (1)
        trans[0, 0] = 0.99; trans[0, 1] = 0.01
        
        # Climb (1) -> Climb (1), Cruise (2), ou Level_Flight (3)
        trans[1, 1] = 0.95; trans[1, 2] = 0.04; trans[1, 3] = 0.01
        
        # Cruise (2) -> Cruise (2) ou Descent (4)
        trans[2, 2] = 0.99; trans[2, 4] = 0.01 
        
        # Level_Flight (3) -> Cruise (2), Descent (4) ou Level (3)
        trans[3, 2] = 0.05; trans[3, 3] = 0.90; trans[3, 4] = 0.05 
        
        # Descent (4) -> Descent (4) ou Ground (0)
        trans[4, 4] = 0.98; trans[4, 0] = 0.02
        return trans

    def fit(self, X, n_iter=250): 
        
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type='diag', 
            n_iter=n_iter,
            tol=1e-3, 
            random_state=42,
            init_params='',
            params='mct', 
            min_covar=1e-3
        )

        self.model.startprob_ = self._create_initial_probabilities()
        self.model.transmat_ = self._create_constrained_transition_matrix()

        # --- MOYENNES (Initialisation basée sur la physique du vol - 5 états) ---
        # [RoC, Groundspeed]
        self.model.means_ = np.array([
            [0, 5],         # Ground_Taxi
            [2500, 350],    # Climb
            [0, 480],       # Cruise
            [0, 300],       # Level_Flight (VR=0, GS intermédiaire)
            [-2000, 250],   # Descent_Approach
        ])
        
        # --- COVARIANCES (Initialisation) ---
        # [Var RoC, Var GS] - La variance est plus faible en Cruise/Level Flight.
        self.model.covars_ = np.array([
            [50**2, 10**2],     # Ground_Taxi: [Var_ROC, Var_GS] - (Forme attendue : (N_états, N_features))
            [1500**2, 100**2],  # Climb
            [500**2, 30**2],    # Cruise
            [1000**2, 75**2],   # Level_Flight
            [1500**2, 100**2],  # Descent_Approach
        ])

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=ConvergenceWarning)
                self.model.fit(X)
                
        except Exception as e:
            # print(f"    ⚠️ Erreur HMM (fit/convergence): {e}")
            return None 
            
        return self

    def predict(self, X):
        states = self.model.predict(X)
        phases = [self.phase_names[s] for s in states]
        return phases


# ============================================
# PIPELINE PRINCIPAL (LOGIQUE 5-PHASES)
# ============================================

def segment_single_flight_5phase(flight_df, flight_id):
    """
    Applique la segmentation HMM 5-états à un DataFrame de vol unique.
    """
    # 1. Vérification de longueur (30 minutes)
    if len(flight_df) < 1800:
        return None, None

    # 2. Préparer les données (Utilise la même fonction de préparation)
    df = prepare_flight_data_3phase_2feature(flight_df)

    # 3. Features pour HMM (doit être [n_samples, 2])
    X = df[['ROC', 'GS']].values
    
    # 4. Entraîner le HMM (Appelle la classe 5-états)
    model = HMM_5State_Robust()
    if model.fit(X) is None:
        return None, None 

    # 5. Prédiction
    phases = model.predict(X)
    
    # 6. Ajouter au DataFrame
    df['phase'] = phases

    # 7. Créer le résumé
    summary = { 'flight_id': flight_id, 'n_points': len(df) }
    for phase in PHASE_ORDER:
        summary[f'n_{phase}'] = (df['phase'] == phase).sum()
    
    return df, summary


# ============================================
# BATCH PROCESSING (À ADAPTER DANS LE MAIN SI NÉCESSAIRE)
# ============================================

def segment_all_flights_from_parquet(parquet_file_path, output_dir, n_flights=None, plot=False):
    """
    Charge un Parquet géant et segmente tous les vols qu'il contient.
    """
    print("="*70)
    print("🚀 SEGMENTATION BATCH (HMM 5-états, 2-Features)")
    print("="*70 + "\n")

    parquet_file = Path(parquet_file_path)
    output_dir = Path(output_dir)
    
    if plot:
        (output_dir / 'plots').mkdir(parents=True, exist_ok=True)

    # 1. Charger le Parquet géant
    print(f"1️⃣  Chargement du Parquet géant: {parquet_file.name} (rapide)")
    try:
        all_data = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"  ⚠️ ERREUR: Impossible de lire le fichier Parquet: {e}")
        return
    print(f"    ✓ {len(all_data)} lignes chargées.\n")

    # 2. Grouper par flight_id
    print("2️⃣  Groupement par flight_id...")
    grouped_flights = all_data.groupby('flight_id')
    n_total_flights = len(grouped_flights)
    print(f"    ✓ {n_total_flights} vols uniques trouvés.\n")

    # 3. Préparer la boucle
    print(f"3️⃣  Démarrage de la segmentation...")
    print(f"    📂 Le dossier de sortie pour les plots/résumés est: {output_dir}\n")
    
    results_summary_list = []
    processed_dfs_list = []
    
    flight_iterator = grouped_flights
    if n_flights:
        flight_iterator = list(grouped_flights)[:n_flights]
        n_total_flights = len(flight_iterator)

    pbar = tqdm(flight_iterator, total=n_total_flights, desc="Vols")

    # 4. Traiter chaque vol
    for flight_id, flight_df in pbar:
        
        pbar.set_description(f"Vol {flight_id}")
        
        try:
            # ⚠️ CHANGEMENT : Appel à la fonction 5-états
            segmented_df, summary = segment_single_flight_5phase(flight_df, flight_id) 

            if segmented_df is None:
                continue
                
            segmented_df = segmented_df.drop(columns=['ROC', 'GS'], errors='ignore')
            processed_dfs_list.append(segmented_df)

            if plot:
                pass # (Fonction de plot non incluse)

            results_summary_list.append(summary)

        except Exception as e:
            print(f"\n⚠️  Erreur majeure sur le vol {flight_id}: {e}")
            continue

    # 5. Combinaison et Sauvegarde finale
    print(f"\n✅ {len(processed_dfs_list)} vols segmentés")
    
    if processed_dfs_list:
        print("5️⃣  Combinaison de tous les DataFrames...")
        final_combined_df = pd.concat(processed_dfs_list, ignore_index=True)
        print("    ✓ DataFrames combinés.")

        # 6. Sauvegarder le Parquet final (rapide)
        output_parquet_path = parquet_file.parent / f"{parquet_file.stem}_5phases_test.parquet" # Changement de nom
        print(f"6️⃣  Sauvegarde du Parquet final : {output_parquet_path}")
        
        final_combined_df.to_parquet(output_parquet_path, index=False)
        print("    ✓ Sauvegarde terminée.")

        # Sauvegarder le résumé
        df_results = pd.DataFrame(results_summary_list)
        summary_file = output_dir / 'segmentation_summary_5phases.csv' # Changement de nom
        df_results.to_csv(summary_file, index=False)
        print(f"📊 Résumé: {summary_file}")
    else:
        print("Aucun vol n'a été traité.")


# ============================================
# MAIN (ADAPTATION)
# ============================================

if __name__ == "__main__":
    
    project_dir = Path.cwd()
    
    default_input_file = project_dir / 'flights_clean_all.parquet'
    default_output_dir = project_dir / 'flights_phases'

    parser = argparse.ArgumentParser(description=f'Segmentation HMM {N_STATES_FINAL}-états (RoC + Groundspeed)')
    
    # ... (Le reste du code main est inchangé) ...
    
    parser.add_argument('--input', type=str, default=str(default_input_file),
                        help=f'Fichier Parquet unique à traiter (défaut: {default_input_file})')
    
    parser.add_argument('--output', type=str, default=str(default_output_dir),
                        help=f'Dossier de sortie (pour plots/résumé) (défaut: {default_output_dir})')
    
    parser.add_argument('--n_flights', type=int, 
                        help='Nombre de vols à traiter (pour test rapide)')
    
    parser.add_argument('--plot', action='store_true',
                        help='Activer la génération de graphiques (un par vol)')

    args = parser.parse_args()
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    segment_all_flights_from_parquet(
        parquet_file_path=args.input, 
        output_dir=args.output, 
        n_flights=args.n_flights,
        plot=args.plot
    )