import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from scipy.stats import mode
from tkinter import *
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
import threading
import matplotlib.colors as mcolors

# ==============================================================================
# --- Data and Model Definitions ---
# ==============================================================================
VALUE_COLS_DD = [
    'Al (Value)', 'Ca (Value)', 'Fe (Value)', 'Mg (Value)',
    'Mn (Value)', 'P (Value)', 'Si (Value)', 'Ti (Value)', 'Unknown (Value)',
    'Dolomite (Value)', 'Goethite (Value)', 'Haematite (Value)', 'Magnetite (Value)',
    'Muscovite (Value)', 'Quartz (Value)', 'Other (Value)', 'Fe-carbonates (Value)'
]
VALUE_COLS_RC = [
    'Al (Value)', 'Ca (Value)', 'Fe (Value)', 'Fe3O4 (Value)', 'Mg (Value)',
    'Mn (Value)', 'P (Value)', 'Si (Value)', 'Ti (Value)', 'Unknown (Value)',
    'Dolomite (Value)', 'Goethite (Value)', 'Haematite (Value)',
    'Muscovite (Value)', 'Quartz (Value)', 'Other (Value)', 'Fe-carbonates (Value)'
]
MODEL_FILES = ['scaler_model_dd.pkl', 'pca_model_dd.pkl', 'kmeans_model_dd.pkl',
               'scaler_model_rc.pkl', 'pca_model_rc.pkl', 'kmeans_model_rc.pkl']

# ==============================================================================
# --- Fused Cluster Mappings and Names ---
# ==============================================================================
# Mapping from DD's arbitrary cluster labels to final reference lithofacies numbers
DD_FINAL_MAPPING = {
    0: 1,
    1: 7,
    2: 4,
    3: 2,
    4: 3,
    5: 1,
    6: 8,
    7: 7,
    8: 8,
    9: 7
}

# Mapping from RC's arbitrary cluster labels to final reference lithofacies numbers
RC_FINAL_MAPPING = {
    0: 2,
    1: 7,
    2: 7,
    3: 1,
    4: 8,
    5: 1,
    6: 7,
    7: 3,
    8: 4,
    9: 8
}

# Mapping from raw clusters (as a tuple of scenario and cluster) to detailed names
DETAILED_LITHOFACIES_MAPPING = {
    ('dd', 0): 'Siliceous iron oxides',
    ('dd', 1): 'Iron oxides, magnetite/goethite subtype',
    ('dd', 2): 'Ferrous silicate',
    ('dd', 3): 'Goethitic iron oxides',
    ('dd', 4): 'Micaceous alumino-silicate, K-rich subtype',
    ('dd', 5): 'Siliceous iron oxides',
    ('dd', 6): 'Carbonates, dolomite-rich',
    ('dd', 7): 'Iron oxides, mixed/clayey',
    ('dd', 8): 'Carbonates, Fe-carbonates subtype',
    ('dd', 9): 'Iron oxides, hematite/goethite',
    
    ('rc', 0): 'Goethitic iron oxides',
    ('rc', 1): 'Iron oxides: goethite/hematite/magnetite',
    ('rc', 2): 'Iron oxides: hematite + magnetite rich',
    ('rc', 3): 'Siliceous iron oxide',
    ('rc', 4): 'Dolomite-rich carbonate',
    ('rc', 5): 'Siliceous ferruginous / manganiferous siliceous oxide',
    ('rc', 6): 'Iron oxides with large “other”/clay admixture',
    ('rc', 7): 'Micaceous alumino-silicate, K-rich / muscovite',
    ('rc', 8): 'Carbonate-ferrous silicate mixed',
    ('rc', 9): 'Fe-carbonates / dolomite-rich carbonate',
}

# Simplified names for the main lithofacies groups
SIMPLIFIED_LITHOFACIES_NAMES = {
    1: 'LF1',
    2: 'LF2',
    3: 'LF3',
    4: 'LF4',
    7: 'LF7',
    8: 'LF8'
}

# Consistent colors for the final lithofacies plots
FINAL_LITHOFACIES_COLORS = {
    1: 'purple',
    2: 'orange',
    3: 'green',
    4: 'yellow',
    7: 'red',
    8: 'cyan'
}


# ==============================================================================
# --- GUI Application Class ---
# ==============================================================================
class GeoClusteringApp:
    def __init__(self, master):
        self.master = master
        master.title("Geochemical Lithofacies Clustering")
        master.geometry("600x600")

        self.FIXED_K_CLUSTERS = 10

        self.create_widgets()

    def create_widgets(self):
        control_frame = Frame(self.master, padx=10, pady=10)
        control_frame.pack(fill=X)

        Label(control_frame, text="Geochemical Lithofacies Clustering Assistant", font=("Helvetica", 16, "bold")).pack(pady=5)
        self.status_label = Label(control_frame, text="Ready", fg="green", font=("Helvetica", 10))
        self.status_label.pack()

        input_frame = Frame(control_frame)
        input_frame.pack(pady=10)
        Label(input_frame, text="New Sample ID:", font=("Helvetica", 10)).pack(side=LEFT, padx=5)
        self.sample_id_entry = Entry(input_frame, width=30)
        self.sample_id_entry.pack(side=LEFT, padx=5)

        button_frame = Frame(control_frame)
        button_frame.pack(pady=10)
        self.train_btn = Button(button_frame, text="Train Models", command=self.run_train_in_thread)
        self.train_btn.pack(side=LEFT, padx=5)
        self.run_btn = Button(button_frame, text="Run Analysis", command=self.run_analysis_in_thread, state=DISABLED)
        self.run_btn.pack(side=LEFT, padx=5)

        self.png_btn = Button(button_frame, text="Generate K-means PNG", command=self.run_png_generation_in_thread, state=DISABLED)
        self.png_btn.pack(side=LEFT, padx=5)
        
        self.force_retrain_var = IntVar()
        self.force_retrain_check = Checkbutton(control_frame, text="Force Retrain", variable=self.force_retrain_var)
        self.force_retrain_check.pack(pady=5)

        self.output_console = ScrolledText(self.master, wrap=WORD, state=DISABLED, width=80, height=20)
        self.output_console.pack(padx=10, pady=10, fill=BOTH, expand=True)

        self.check_initial_state()
        self.sample_id_entry.bind("<KeyRelease>", self.check_entry)
        
    def check_initial_state(self):
        if all(os.path.exists(f) for f in MODEL_FILES):
            self.run_btn.config(state=NORMAL)
            self.png_btn.config(state=NORMAL)
            self.output("Existing models found. You can run an analysis directly.")
        else:
            self.output("No models found. Please train models first.")
            
    def check_entry(self, event=None):
        if self.sample_id_entry.get().strip() != "":
            self.run_btn.config(state=NORMAL)
        else:
            self.run_btn.config(state=DISABLED)

    def output(self, text):
        self.output_console.config(state=NORMAL)
        self.output_console.insert(END, text + "\n")
        self.output_console.see(END)
        self.output_console.config(state=DISABLED)

    def set_status(self, text, color="black"):
        self.status_label.config(text=text, fg=color)
        self.master.update_idletasks()

    def run_train_in_thread(self):
        self.set_status("Training models... Please wait.")
        self.train_btn.config(state=DISABLED)
        self.run_btn.config(state=DISABLED)
        self.png_btn.config(state=DISABLED)
        self.output_console.config(state=NORMAL)
        self.output_console.delete(1.0, END)
        self.output_console.config(state=DISABLED)
        
        thread = threading.Thread(target=self.part1_train_models)
        thread.start()

    def run_analysis_in_thread(self):
        sample_id = self.sample_id_entry.get().strip()
        if not sample_id:
            messagebox.showwarning("Input Error", "Please enter a Sample ID.")
            return

        self.set_status("Running analysis... Please wait.")
        self.train_btn.config(state=DISABLED)
        self.run_btn.config(state=DISABLED)
        self.png_btn.config(state=DISABLED)
        self.output_console.config(state=NORMAL)
        self.output_console.delete(1.0, END)
        self.output_console.config(state=DISABLED)

        thread = threading.Thread(target=self.part2_and_3_apply_model, args=(sample_id,))
        thread.start()
    
    def run_png_generation_in_thread(self):
        self.set_status("Generating K-means PNG... Please wait.")
        self.train_btn.config(state=DISABLED)
        self.run_btn.config(state=DISABLED)
        self.png_btn.config(state=DISABLED)
        self.output_console.config(state=NORMAL)
        self.output_console.delete(1.0, END)
        self.output_console.config(state=DISABLED)

        thread = threading.Thread(target=self.generate_k_means_png)
        thread.start()

    def generate_k_means_png(self):
        self.output("--- GENERATING K-MEANS ELBOW PLOTS ---")
        try:
            df_full = pd.read_csv('merged_geochemical_data.csv')
        except FileNotFoundError:
            self.output("Error: 'merged_geochemical_data.csv' not found.")
            self.set_status("PNG Generation Failed", "red")
            self.train_btn.config(state=NORMAL)
            self.run_btn.config(state=NORMAL)
            self.png_btn.config(state=NORMAL)
            return

        df_full['Fe-carbonates (Value)'] = df_full['Ankerite (Value)'] + df_full['Siderite (Value)']
        
        X_dd = df_full[VALUE_COLS_DD].copy()
        X_rc = df_full[VALUE_COLS_RC].copy()
        
        self.determine_best_k(X_dd, 'dd', fixed_k=13)
        self.determine_best_k(X_rc, 'rc', fixed_k=14)

        self.set_status("PNG Generation Complete", "green")
        self.train_btn.config(state=NORMAL)
        self.run_btn.config(state=NORMAL)
        self.png_btn.config(state=NORMAL)

    def determine_best_k(self, X_train, scenario, fixed_k=None):
        self.output(f"\nDetermining best 'k' for the '{scenario}' scenario...")
        inertias = []
        range_k = range(2, 41)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        pca_temp = PCA()
        X_train_pca = pca_temp.fit_transform(X_train_scaled)

        for k in range_k:
            kmeans_model = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_train_pca)
            inertias.append(kmeans_model.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range_k, inertias, marker='o', linestyle='-', color='b')
        plt.title(f'Elbow Method for Optimal K ({scenario})')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia')
        plt.grid(True)
        
        if fixed_k is not None:
            best_k = fixed_k
            best_k_index = list(range_k).index(best_k)
        else:
            if len(inertias) > 2:
                second_derivatives = np.diff(inertias, n=2)
                best_k_index = np.argmin(second_derivatives) + 2
                best_k = range_k[best_k_index]
            else:
                best_k = range_k[0]
                best_k_index = 0

        plt.axvline(x=best_k, color='r', linestyle='--', label=f'Suggested Optimal K: {best_k}')
        plt.text(best_k, inertias[best_k_index], f'  Optimal K: {best_k}', ha='right', va='top', color='red')
        plt.legend()
        self.output(f"Suggested optimal K for {scenario}: {best_k}")
        
        plt.savefig(f'kmeans_elbow_method_{scenario}.png')
        plt.close()
        self.output(f"'kmeans_elbow_method_{scenario}.png' generated successfully.")

    # ==============================================================================
    # --- PART 1: Model Training Function ---
    # ==============================================================================
    def part1_train_models(self):
        self.output("--- TRAINING AND SAVING MODELS ---")
        try:
            df_full = pd.read_csv('merged_geochemical_data.csv')
        except FileNotFoundError:
            self.output("Error: 'merged_geochemical_data.csv' not found.")
            self.set_status("Training Failed", "red")
            self.train_btn.config(state=NORMAL)
            self.run_btn.config(state=NORMAL)
            self.png_btn.config(state=NORMAL)
            return

        df_full['Fe-carbonates (Value)'] = df_full['Ankerite (Value)'] + df_full['Siderite (Value)']
        
        X_dd = df_full[VALUE_COLS_DD].copy()
        y_dd = df_full['HOLE_ID']
        X_train_dd, X_test_dd, _, _ = train_test_split(X_dd, y_dd, test_size=0.2, random_state=42)

        X_rc = df_full[VALUE_COLS_RC].copy()
        y_rc = df_full['HOLE_ID']
        X_train_rc, X_test_rc, _, _ = train_test_split(X_rc, y_rc, test_size=0.2, random_state=42)

        try:
            self.train_and_save('dd', X_train_dd, X_test_dd)
            self.train_and_save('rc', X_train_rc, X_test_rc)
            self.set_status("Training Complete", "green")
        except Exception as e:
            self.output(f"An error occurred during training: {e}")
            self.set_status("Training Failed", "red")
        
        self.train_btn.config(state=NORMAL)
        self.run_btn.config(state=NORMAL)
        self.png_btn.config(state=NORMAL)

    def train_and_save(self, scenario, X_train, X_test):
        self.output(f"\nTraining models for the '{scenario}' scenario...")
        value_cols = VALUE_COLS_DD if scenario == 'dd' else VALUE_COLS_RC
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        pca_temp = PCA()
        pca_temp.fit(X_train_scaled)
        cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.where(cumulative_variance >= 0.90)[0][0] + 1
        self.output(f"Number of PCA components to explain 90% variance: {n_components}")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        plt.title(f'Cumulative Explained Variance by Principal Components ({scenario})')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Threshold')
        plt.axvline(x=n_components, color='g', linestyle='--', label=f'{n_components} components for 90% variance')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'pca_explained_variance_{scenario}.png')
        plt.close()
        self.output(f"PCA explained variance plot saved as 'pca_explained_variance_{scenario}.png'")

        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)

        k_final = self.FIXED_K_CLUSTERS
        self.output(f"Using a fixed K for clustering: {k_final}")
        
        kmeans = KMeans(n_clusters=k_final, random_state=42, n_init='auto')
        kmeans.fit(X_train_pca)
        
        X_test_scaled = scaler.transform(X_test)
        X_test_pca = pca.transform(X_test_scaled)
        test_clusters = kmeans.predict(X_test_pca)
        silhouette_avg = silhouette_score(X_test_pca, test_clusters)
        self.output(f"The Silhouette Coefficient on the test set ({scenario}) is: {silhouette_avg:.4f}")

        self.output("\n--- GENERATING CLUSTER CHARACTERISTICS ---")
        temp_df_original = pd.DataFrame(
            columns=value_cols,
            data=scaler.inverse_transform(pca.inverse_transform(kmeans.cluster_centers_))
        )
        centroids_df = temp_df_original.copy()
        centroids_df.index.name = 'Cluster'
        with open(f'cluster_characteristics_{scenario}.txt', 'w') as f:
            f.write("Cluster Characteristics (Lithofacies) based on original mean values:\n\n")
            for i in range(len(centroids_df)):
                f.write(f"--- Cluster {i} ---\n")
                cluster_data = centroids_df.iloc[i].sort_values(ascending=False)
                for feature, value in cluster_data.items():
                    f.write(f"  {feature}: {value:.4f}\n")
                f.write("\n")
        self.output(f"'cluster_characteristics_{scenario}.txt' file generated.")

        joblib.dump(scaler, f'scaler_model_{scenario}.pkl')
        joblib.dump(pca, f'pca_model_{scenario}.pkl')
        joblib.dump(kmeans, f'kmeans_model_{scenario}.pkl')
        self.output(f"Models for {scenario.upper()} saved successfully.")


    # ==============================================================================
    # --- PART 2 & 3: Application Function ---
    # ==============================================================================
    def part2_and_3_apply_model(self, new_sample_id):
        self.output(f"\n--- PROCESSING NEW SAMPLE: {new_sample_id} ---")
        try:
            df_elements = pd.read_csv(f'CNNelementsprediction_{new_sample_id}.csv')
            df_minerals = pd.read_csv(f'CNNmineralsprediction_{new_sample_id}.csv')
            df_new_sample = pd.merge(df_elements, df_minerals, on=['HOLE_ID', 'Borehole Depth [m]', 'Logged Depth [m]', 'Source'], how='inner')
        except FileNotFoundError as e:
            self.output(f"Error: One of the new sample files was not found: {e}.")
            self.set_status("Analysis Failed", "red")
            self.train_btn.config(state=NORMAL)
            self.run_btn.config(state=NORMAL)
            self.png_btn.config(state=NORMAL)
            return

        df_new_sample['Fe-carbonates (Value)'] = df_new_sample['Ankerite (Value)'] + df_new_sample['Siderite (Value)']

        if 'RC' in new_sample_id:
            self.output("RC sample detected. Loading 'RC' models.")
            scenario = 'rc'
            value_cols_current = VALUE_COLS_RC
            final_mapping = RC_FINAL_MAPPING
        else:
            self.output("Non-RC (DD) sample detected. Loading 'DD' models.")
            scenario = 'dd'
            value_cols_current = VALUE_COLS_DD
            final_mapping = DD_FINAL_MAPPING
        
        try:
            scaler_loaded = joblib.load(f'scaler_model_{scenario}.pkl')
            pca_loaded = joblib.load(f'pca_model_{scenario}.pkl')
            kmeans_loaded = joblib.load(f'kmeans_model_{scenario}.pkl')
        except FileNotFoundError:
            self.output(f"Error: Models for scenario '{scenario}' were not found.")
            self.output("Please run the script to train models first.")
            self.set_status("Analysis Failed", "red")
            self.train_btn.config(state=NORMAL)
            self.run_btn.config(state=NORMAL)
            self.png_btn.config(state=NORMAL)
            return
        
        missing_cols = [col for col in value_cols_current if col not in df_new_sample.columns]
        if missing_cols:
            self.output(f"Error: Missing columns for the current sample: {missing_cols}")
            self.set_status("Analysis Failed", "red")
            self.train_btn.config(state=NORMAL)
            self.run_btn.config(state=NORMAL)
            self.png_btn.config(state=NORMAL)
            return

        X_new = df_new_sample[value_cols_current].copy()
        X_new_scaled = scaler_loaded.transform(X_new)
        X_new_pca = pca_loaded.transform(X_new_scaled)
        
        # Predict initial clusters (0-9)
        df_new_sample['raw_cluster'] = kmeans_loaded.predict(X_new_pca)
        
        # Map the raw clusters to the final, fused lithofacies numbers
        df_new_sample['lithofacies_cluster'] = df_new_sample['raw_cluster'].map(final_mapping)
        
        # Map to the detailed names for the final output
        df_new_sample['lithofacies_name_detailed'] = df_new_sample.apply(
            lambda row: DETAILED_LITHOFACIES_MAPPING.get((scenario, row['raw_cluster'])),
            axis=1
        )

        self.output("\nInitial lithofacies identified:")
        self.output(df_new_sample.groupby('lithofacies_name_detailed').size().to_string())

        self.output("\n--- APPLYING CONDITIONAL SMOOTHING ---")
        window_size = 30 if 'RC' in new_sample_id else 5000
        self.output(f"Using a smoothing window size of {window_size} for {scenario.upper()} sample.")
        
        smoothed_clusters = df_new_sample['lithofacies_cluster'].rolling(
            window=window_size, center=True, min_periods=1
        ).apply(lambda x: mode(x)[0], raw=True)
        df_new_sample['lithofacies_cluster_smoothed'] = smoothed_clusters.astype(int)
        df_new_sample['lithofacies_name_smoothed'] = df_new_sample['lithofacies_cluster_smoothed'].map(SIMPLIFIED_LITHOFACIES_NAMES)

        self.output("Lithofacies after smoothing:")
        self.output(df_new_sample.groupby('lithofacies_name_smoothed').size().to_string())

        self.output("\n--- GENERATING LITHOFACIES CSV ---")
        litofacies_blocks_smoothed = []
        if not df_new_sample.empty:
            df_sorted = df_new_sample.sort_values(by='Borehole Depth [m]').reset_index(drop=True)
            current_name = df_sorted['lithofacies_name_smoothed'].iloc[0]
            start_depth = df_sorted['Borehole Depth [m]'].iloc[0]
            hole_id = df_sorted['HOLE_ID'].iloc[0]
            for i in range(1, len(df_sorted)):
                next_name = df_sorted['lithofacies_name_smoothed'].iloc[i]
                if next_name != current_name:
                    end_depth = df_sorted['Borehole Depth [m]'].iloc[i-1]
                    litofacies_blocks_smoothed.append({
                        'HOLE_ID': hole_id, 'From [m]': start_depth, 'To [m]': end_depth, 'Cluster': current_name
                    })
                    current_name = next_name
                    start_depth = df_sorted['Borehole Depth [m]'].iloc[i]
            end_depth = df_sorted['Borehole Depth [m]'].iloc[-1]
            litofacies_blocks_smoothed.append({
                'HOLE_ID': hole_id, 'From [m]': start_depth, 'To [m]': end_depth, 'Cluster': current_name
            })
        
        df_litofacies_blocks = pd.DataFrame(litofacies_blocks_smoothed)
        output_lf_file = f'{new_sample_id}_LF.csv'
        df_litofacies_blocks.to_csv(output_lf_file, index=False)
        self.output(f"Lithofacies block file '{output_lf_file}' generated.")

        self.set_status("Analysis Complete", "green")
        self.train_btn.config(state=NORMAL)
        self.run_btn.config(state=NORMAL)
        self.png_btn.config(state=NORMAL)
        self.part4_visualize(df_new_sample, new_sample_id, scenario)
        
    def part4_visualize(self, df_new_sample, new_sample_id, scenario):
        self.output("\n--- GENERATING VISUALIZATION ---")
        try:
            df_sorted = df_new_sample.sort_values(by='Borehole Depth [m]').reset_index(drop=True)
            unique_clusters = sorted(df_sorted['lithofacies_cluster_smoothed'].unique())
            
            cluster_name_to_color = {
                SIMPLIFIED_LITHOFACIES_NAMES[c]: FINAL_LITHOFACIES_COLORS.get(c, 'gray') for c in unique_clusters
            }
            
            fig, ax = plt.subplots(figsize=(6, 12))
            ax.set_ylim(df_sorted['Borehole Depth [m]'].min(), df_sorted['Borehole Depth [m]'].max())
            ax.invert_yaxis()
            ax.set_title(f'Lithofacies for Sample {new_sample_id}')
            ax.set_xlabel('Lithofacies')
            ax.set_ylabel('Depth [m]')
            ax.set_xticks([])

            if not df_sorted.empty:
                start_depth = df_sorted['Borehole Depth [m]'].iloc[0]
                current_name = df_sorted['lithofacies_name_smoothed'].iloc[0]
                
                for i in range(1, len(df_sorted)):
                    next_depth = df_sorted['Borehole Depth [m]'].iloc[i]
                    next_name = df_sorted['lithofacies_name_smoothed'].iloc[i]
                    if next_name != current_name:
                        color = cluster_name_to_color.get(current_name, 'gray')
                        ax.axhspan(start_depth, next_depth, facecolor=color, edgecolor='black', linewidth=0.5)
                        ax.text(0.5, (start_depth + next_depth) / 2, current_name,
                                transform=ax.get_yaxis_transform(), ha='center', va='center',
                                fontsize=8, color='black', weight='bold')
                        current_name = next_name
                        start_depth = next_depth
                
                color = cluster_name_to_color.get(current_name, 'gray')
                ax.axhspan(start_depth, df_sorted['Borehole Depth [m]'].iloc[-1], facecolor=color, edgecolor='black', linewidth=0.5)
                ax.text(0.5, (start_depth + df_sorted['Borehole Depth [m]'].iloc[-1]) / 2, current_name,
                        transform=ax.get_yaxis_transform(), ha='center', va='center',
                        fontsize=8, color='black', weight='bold')

            handles = [plt.Rectangle((0,0),1,1, color=color, label=name) for name, color in cluster_name_to_color.items()]
            ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1))

            plt.tight_layout()
            output_plot_file = f'lithofacies_plot_{new_sample_id}.png'
            plt.savefig(output_plot_file, bbox_inches='tight')
            plt.close()
            self.output(f"Lithofacies plot saved as '{output_plot_file}'")

        except Exception as e:
            self.output(f"Error generating plot: {e}")

# ==============================================================================
# --- Main Execution ---
# ==============================================================================
if __name__ == "__main__":
    root = Tk()
    app = GeoClusteringApp(root)
    root.mainloop()
