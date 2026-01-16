import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm, poisson, skew, kurtosis



class LsLabQC():
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'",
            self.conn
        )

        # Wichtige Daten
        self.df_info = pd.read_sql_query("SELECT * FROM RepDLSSizing", self.conn) # basis daten. Measuremtns, reps etc.
        self.df_raw_data = pd.read_sql_query("SELECT * FROM RepetitionRawData", self.conn) # die Rohdaten sind in der RepetitionRawData Tabelle... Nach langem suchen :/
        self.df_meas_settings_data = pd.read_sql_query("SELECT * FROM MeasurementRawData", self.conn) # scatter angle

        self.median_count_deviation_max = 1.5
        self.skew_max = 1.0
        self.kurt_max = 5
        self.nr_intercept_calc_values = 16

    def list_all_tables(self):
        with pd.option_context("display.max_rows", None, "display.max_colwidth", None, "display.max_columns", None, "display.width", 0):
            print(self.tables) 

    def read_table(self, table_nr):
        table = self.tables.loc[table_nr, 'name']
        columns = pd.read_sql_query(f"PRAGMA table_info('{table}')", self.conn)
        
        with pd.option_context("display.max_rows", None, "display.max_colwidth", None, "display.max_columns", None, "display.width", 0):
            print(f"Columns for table {table_nr}, {table}':")
            print(columns[['name', 'type']])

    def get_meas_reps(self):
        reps_per_meas = self.df_info.groupby('MeasDLSSizingBase_id').size().tolist()
        return reps_per_meas
    
    def get_scatter_angle(self, meas_idx):
        angle = self.df_meas_settings_data.loc[meas_idx, "ScatteringAngleInDegree"]
        return angle 
    
    def build_rep_index_map(self):
        meas_reps = self.get_meas_reps()
        mapping = []
        for meas_idx, n_rep in enumerate(meas_reps):
            for rep_idx in range(n_rep):
                mapping.append((meas_idx, rep_idx))
        return mapping
    
    def get_count_rate(self,i):
        self.counts = self.df_raw_data.loc[i, "CountTraceABytes"]
        arr = np.frombuffer(self.counts, dtype=float)
        return(arr)

    def flag_data_evil_skew(self, i):
        # wenn die normalverteilung der countrate geskewd ist, ist es staubig...  
        counts = self.get_count_rate(i)
        sk = skew(counts)
        ku = kurtosis(counts, fisher=False)

        is_dust = (sk > self.skew_max) or (ku > self.kurt_max)

        return is_dust
    
    def get_acf_data(self, i):
        self.blob_g2_t = self.df_raw_data.loc[i, "CorrelationFunctionBytes"] 
        arr = np.frombuffer(self.blob_g2_t, dtype=np.float64) # das array hat die struktur: gerade werde: g2, ungerade Werte: t
        self.t = arr[0::2]
        self.g2 = arr[1::2]
        return self.t, self.g2
    
    def flag_acf_intercept(self, i):
        # linfit über die ersten Werte der Acf, wenn der schnitt mit y-achse über 1 ist ciao
        t, g2 = self.get_acf_data(i)
        nr = self.nr_intercept_calc_values
        x = t[:nr]
        y = g2[:nr]
        k, d = np.polyfit(x, y, 1)

        is_fcked = d > 1.0
        return is_fcked

    def flag_deviation_from_median_Intensity(self, i):
        # selbsterklärend
        counts = self.get_count_rate(i)
        median_intensity = np.median(counts)
        upper_limit = median_intensity * self.median_count_deviation_max

        is_dust = np.any(counts > upper_limit)
        return is_dust
        

    def build_qc_map(self):
        meas_reps = self.get_meas_reps()
        rep_map = self.build_rep_index_map()

        qc_map = {}

        for global_idx, (meas_idx, rep_idx) in enumerate(rep_map):

            is_dust_skew = self.flag_data_evil_skew(global_idx)
            is_dust_deviation = self.flag_deviation_from_median_Intensity(global_idx)
            is_fcked = self.flag_acf_intercept(global_idx)
            scat_angl = self.get_scatter_angle(meas_idx)

            flags = []
            if is_dust_skew:
                flags.append("dust_skew")
            if is_dust_deviation:
                flags.append("dust_deviation")
            if is_fcked:
                flags.append("acf_fcked")

            qc_map.setdefault(meas_idx, {
                "n_reps": meas_reps[meas_idx],
                "n_flagged": 0,
                "scatter angle":float(scat_angl),
                "reps": {}
            })

            qc_map[meas_idx]["reps"][rep_idx] = {
                "global_idx": global_idx,
                "status": "bad" if flags else "passt",
                "flags": flags
            }

            if flags:
                qc_map[meas_idx]["n_flagged"] += 1

        self.qc_map = qc_map
        return qc_map
    
    def get_unflagged_global_mask(self):
        qc_map = self.qc_map
        global_mask = [rep_data['global_idx']
               for meas_data in qc_map.values()
               for rep_data in meas_data['reps'].values()
               if rep_data['status'] == 'passt']
        
        return global_mask

    def get_unflagged_mask_for_meas(self, meas_idx):
        qc_map = self.qc_map

        if meas_idx not in qc_map:
            raise ValueError(f"Measurement {meas_idx} not in QC map.")

        local_mask = [
            rep_data['global_idx']
            for rep_data in qc_map[meas_idx]['reps'].values()
            if rep_data['status'] == 'passt'
        ]
        return local_mask
    
    def get_all_mask_for_meas(self, meas_idx):
        qc_map = self.qc_map

        if meas_idx not in qc_map:
            raise ValueError(f"Measurement {meas_idx} not in QC map.")

        local_mask = [
            rep_data['global_idx']
            for rep_data in qc_map[meas_idx]['reps'].values()
        ]
        return local_mask
    
    def get_unflagged_mask_for_multi_meas(self, meas_idxs):
        # eg. meas_idx = list

        qc_map = self.qc_map

        for meas_idx in meas_idxs:
            if meas_idx not in qc_map:
                raise ValueError(f"Measurement {meas_idx} not in QC map.")

        mask = [
            [rep_data['global_idx'] 
            for rep_data in qc_map[m_idx]['reps'].values() 
            if rep_data['status'] == 'passt'
            ]
            for m_idx in meas_idxs
            ]

        return mask

    
    def plot_hist_counts_per_meas(self, meas):
        reps = self.get_all_mask_for_meas(meas)
        bins=30
        n_reps = len(reps)

            
        fig, axes = plt.subplots(1, n_reps, figsize=(6 * n_reps, 4))
        if n_reps == 1:
            axes = [axes] 

        for ax, i in zip(axes, reps):
            counts = self.get_count_rate(i)
            frequency, bin_edges = np.histogram(counts, bins=bins)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            ax.bar(bin_centers, frequency, width=np.diff(bin_edges), edgecolor='black')
            ax.set_xlabel("Count Rate Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Repetition {i}")

            mu = np.mean(counts)
            sigma = np.std(counts)

            x = np.linspace(np.min(counts), np.max(counts), 500)
            bin_width = bin_edges[1] - bin_edges[0]
            gauss = norm.pdf(x, mu, sigma) * len(counts) * bin_width # es ist fix nicht gauß verteilt... Aber ich bin kein stat Genie und es reicht für unsere Zwecke 

            ax.plot(
                x,
                gauss,
                'r-',
                lw=2,
                label=f'Gaussian fit ($\mu$={mu:.1f}, $\sigma$={sigma:.1f})'
            )

            median_val = np.median(counts)
            threshold = median_val * self.median_count_deviation_max
            ax.axvline(threshold, color='green', linestyle='--', lw=2, 
                       label=f'Threshold ({self.median_count_deviation_max}x Median)')
            
            ax.legend()
                
        plt.tight_layout()
        plt.show()






class LsLabAnalyze():
    def __init__(self, db_path, mask = None):
        self.conn = sqlite3.connect(db_path)
        self.tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'",
            self.conn
        )

        # Wichtige Daten
        self.df_info = pd.read_sql_query("SELECT * FROM RepDLSSizing", self.conn) # basis daten. Measuremtns, reps etc.
        self.df_raw_data = pd.read_sql_query("SELECT * FROM RepetitionRawData", self.conn) # die Rohdaten sind in der RepetitionRawData Tabelle... Nach langem suchen :/
        self.df_fit_data = pd.read_sql_query("SELECT * FROM Cumulant", self.conn) # cum bytes
        self.df_cumulant = pd.read_sql_query("SELECT * FROM CumulantFitValue", self.conn) #cumulants
        self.df_refr_data = pd.read_sql_query("SELECT * FROM RepDLSSizingCumulantResult", self.conn) #n und eta
        self.df_meas_settings_data = pd.read_sql_query("SELECT * FROM MeasurementRawData", self.conn) # scatter angle
        self.df_laser_data = pd.read_sql_query("SELECT * FROM Laser", self.conn) # Wellenlänge
        
        if mask is None:
            nr_reps = sum(self.df_info.groupby('MeasDLSSizingBase_id').size().tolist()) 
            self.rep_idx = np.arange(0, nr_reps)
            self.update_meas_reps = False
        else:
            self.mask = mask
            self.rep_idx = [idx for sublist in mask for idx in sublist]
            self.update_meas_reps = True
            self.nr_all_reps = sum(self.df_info.groupby('MeasDLSSizingBase_id').size().tolist()) 

        self.sql_query_gamma_and_std()
        

    def list_all_tables(self):
        with pd.option_context("display.max_rows", None, "display.max_colwidth", None, "display.max_columns", None, "display.width", 0):
            print(self.tables) 

    def read_table(self, table_nr):
        table = self.tables.loc[table_nr, 'name']
        columns = pd.read_sql_query(f"PRAGMA table_info('{table}')", self.conn)
        
        with pd.option_context("display.max_rows", None, "display.max_colwidth", None, "display.max_columns", None, "display.width", 0):
            print(f"Columns for table {table_nr}, {table}':")
            print(columns[['name', 'type']])

    def get_meas_reps(self):
        if not self.update_meas_reps:
            reps_per_meas = self.df_info.groupby('MeasDLSSizingBase_id').size().tolist()
            return reps_per_meas # z.B [1,3,3,3] erste messung 1 rep andere 3 messungen 3 reps
        if self.update_meas_reps:
            reps_per_meas = [len(sublist) for sublist in self.mask]
            return reps_per_meas


    def get_acf_data(self, i):
        self.blob_g2_t = self.df_raw_data.loc[i, "CorrelationFunctionBytes"] 
        arr = np.frombuffer(self.blob_g2_t, dtype=np.float64) # das array hat die struktur: ungerade werde: g2, gerade Werte: t
        self.t = arr[0::2]
        self.g2 = arr[1::2]
        return self.t, self.g2

    def plot_setup(self, size=(6, 4)):
        fig, ax = plt.subplots(figsize=size)
        ax.tick_params(axis='both', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        return fig, ax
    

    def plot_all_acf(self, size=(6, 4)):
        fig, ax = self.plot_setup(size=size)
        ax.set_xlabel(r"$t' \, (\mathrm{ns})$")
        ax.set_ylabel(r"$g^{(2)}$")
        ax.set_xscale('log')

        # Pastel colormap
        cmap = plt.cm.Set3

        reps_per_meas = self.get_meas_reps()
        global_idx = 0
        n_meas = len(reps_per_meas)

        for meas_idx, n_reps in enumerate(reps_per_meas):
            color = cmap(meas_idx % cmap.N)
            scatter_angle = self.get_scatter_angle(meas_idx)

            for rep_local_idx in range(n_reps):
                rep_global_idx = self.rep_idx[global_idx]
                global_idx += 1

                t, g2 = self.get_acf_data(rep_global_idx)

                label = None
                if rep_local_idx == 0:
                    label = f"Meas {meas_idx} (θ = {scatter_angle:.1f}°)"

                ax.plot(t, g2, marker='o', linestyle='', markersize=2, color=color, alpha=0.8, markerfacecolor='none', label=label)

        ax.legend(fontsize=7, frameon=False)
        plt.tight_layout()
        plt.savefig("/home/elias/proj/_dls_sql/acf_all.png", dpi=300, bbox_inches="tight")
        plt.show()

    def sql_query_gamma_and_std(self):
        #die sql query ist gevibed. Ich kann das nicht ordentlich und bis jetzt konnte ich wirkliches sql nutzen vermeiden haha
        query = """
        SELECT rep.RepetitionDLSBase_id, fit.K1, fit.K2, fit.K3
        FROM RepDLSSizing rep
        JOIN RepDLSSizingCumulantSettings settings ON rep.RepetitionDLSBase_id = settings.RepDLSSizing_id
        JOIN CumulantDLSSizingProcessingPair pair ON settings.Id = pair.Settings_id
        JOIN RepDLSSizingCumulantResult res ON pair.Result_id = res.Id
        JOIN CumulantDLSSizing sizing ON res.Id = sizing.RepDLSSizingCumulantResultId
        JOIN Cumulant c ON sizing.CumulantBase_id = c.Id
        JOIN CumulantFitValue fit ON c.CumulantFitValue_id = fit.Id
        """
        df = pd.read_sql_query(query, self.conn)

        # 2. Filter for the 2nd Order (Quadratic) fit results only
        df_filtered = df[df['K2'].notna() & df['K3'].isna()].copy()

        self.gamma_map = {
        row['RepetitionDLSBase_id']: (row['K1'], row['K2'])
        for _, row in df_filtered.iterrows()
        }
    

    def get_gamma_and_std(self, rep_idx):
        rep_id = self.df_info.loc[rep_idx, 'RepetitionDLSBase_id']

        if rep_id not in self.gamma_map:
            raise KeyError(f"No cum {rep_id}")

        k1, k2 = self.gamma_map[rep_id]
        gamma = k1
        std = np.sqrt(k2)
        return gamma, std
            
    def get_all_gamma_and_std(self):
        gamma_map = {}
        std_map = {}

        for i in self.rep_idx:
            gamma, std = self.get_gamma_and_std(i)
            gamma_map[i] = gamma
            std_map[i] = std

        return gamma_map, std_map

    def get_T(self):
        temp_K = self.df_raw_data.loc[0, 'TemperatureInCelsius'] + 273.15
        return temp_K
    
    def get_eta(self):
        eta = self.df_refr_data.loc[0, "Viscosity"] * 1e-3
        return eta

    def get_scatter_angle(self, i):
        angle = self.df_meas_settings_data.loc[i, "ScatteringAngleInDegree"]
        return angle 
    
    def get_wavelenght(self):
        wavelenght = self.df_laser_data.loc[0, "WaveLengthAccess"]
        return wavelenght

    def get_refractive_index(self):
        n = self.df_refr_data.loc[0, "RefractiveIndex"]
        return n

    def get_q(self, meas_idx):
        wavelenght = self.get_wavelenght()*1e-9
        n = self.get_refractive_index()
        angle = self.get_scatter_angle(meas_idx) * (np.pi /180)

        q = ((4*np.pi*n)/wavelenght) * np.sin(angle/2) 
        return q
    

    def get_all_q(self):
        all_q = []
        nr_meas = len(self.get_meas_reps())
        for meas_idx in range(nr_meas):
            q = float(self.get_q(meas_idx))
            all_q.append(q)

        return all_q
    
    def get_avg_gamma_per_meas(self):
        # die funktion berechnet durschnittliches gamma pro messung (z.B. 3 reps pro messung) und
        # berechnet die unsicherheit aus der stat. unsicherheit des mittelwerts und unsicherheit des fits
        gamma_map, std_map = self.get_all_gamma_and_std()
        mask = self.mask  # e.g. [[1,2,3], [4,5,6]]

        all_avg_gamma = []
        all_std_gamma = []

        for rep_arr in mask:
            # Extract values safely via dicts
            gammas_per_meas = np.array([gamma_map[i] for i in rep_arr])
            fits_std_per_meas = np.array([std_map[i] for i in rep_arr])
            n = len(gammas_per_meas)
            avg_gamma = np.mean(gammas_per_meas)
            all_avg_gamma.append(avg_gamma)

            if n > 1:
                sigma_stat = np.std(gammas_per_meas, ddof=1) / np.sqrt(n) 
                sigma_mean_fit = np.sqrt(np.sum(np.square(fits_std_per_meas))) / n
                std_gamma = np.sqrt(sigma_stat**2 + sigma_mean_fit**2)
            else:
                std_gamma = fits_std_per_meas[0] # nur die unsicherheit vom fit bei einzelmessung
            all_std_gamma.append(std_gamma)


    
        all_avg_gamma = np.array(all_avg_gamma)
        all_std_gamma = np.array(all_std_gamma)


        return all_avg_gamma, all_std_gamma
    


    
    def fit_gamma_q(self):
        qs = np.array(self.get_all_q())

        x_vals = qs**2
        y_vals, y_std = self.get_avg_gamma_per_meas()

        lin_fit = lambda x, D: D * x

        try:
            self.popt, self.pcov = curve_fit(lin_fit, x_vals, y_vals, p0=[1], sigma=y_std, absolute_sigma=True)
        except RuntimeError as e:
            print(f"Curve fitting failed: {e}")
            return
        
        D_fit = self.popt[0]
        D_std = np.sqrt(self.pcov[0, 0])
        x_fit = np.linspace(0, np.max(x_vals) * 1.05, 100)
        y_fit = lin_fit(x_fit,D_fit)

        fig, ax = self.plot_setup()
        ax.set_xlabel(r"$q^2$")
        ax.set_ylabel(r"$\Gamma$")

        ax.errorbar(x_vals, y_vals, yerr=y_std, fmt='.', capsize=3, label='Avg. gamma per Meas.')
        ax.plot(x_fit, y_fit, linestyle='--', color='red', label=f'D = {D_fit:.4e} \pm {D_std:.4e}$')
        ax.legend()
        plt.savefig("/home/elias/proj/_dls_sql/gamma_vs_q.png", dpi=300, bbox_inches="tight")
        plt.show()

    def get_D(self):
        qs = np.array(self.get_all_q())
        gammas, y_std = self.get_avg_gamma_per_meas()

        x_vals = qs**2
        y_vals = gammas
        lin_fit = lambda x, D: D * x

        try:
            self.popt, self.pcov = curve_fit(lin_fit, x_vals, y_vals, p0=[1], sigma=y_std, absolute_sigma=True)
        except RuntimeError as e:
            print(f"Curve fitting failed: {e}")
            return
        
        D_fit = self.popt[0]
        D_std = np.sqrt(self.pcov[0, 0])
        return D_fit, D_std
    
    def get_particle_radius(self, viscosity = None):
        temp = self.get_T()
        D, D_std = self.get_D()
        k_B = 1.380649e-23 

        if viscosity is None:
            eta = self.get_eta()
        else:
            eta = viscosity

        R = (k_B * temp) / (6* np.pi * eta * D)

        R_std = R * (D_std / D)
        return float(R), float(R_std)

    


    





def test_everything(meas, path):
    #quality checker
    meas_qc = LsLabQC(path)
    meas_qc.build_qc_map
    from pprint import pprint
    pprint(meas_qc.build_qc_map())
    print(meas_qc.get_unflagged_mask_for_multi_meas([0,1,2]))
    meas_qc.plot_hist_counts_per_meas(meas)
    #meas_qc.bs_poisson(meas)

    #analyzer
    mask = meas_qc.get_unflagged_mask_for_meas(meas)
    meas = LsLabAnalyze(path, mask = mask)
    meas.plot_all_acf()
    print(meas.get_all_gamma())




#test_everything(3, '/home/elias/proj/_dls_sql/EP_Sicastar_04122025.lslab2')

meas_qc = LsLabQC("/home/elias/proj/_dls_sql/markus_hfip_water_salt.lslab2")
meas_qc.build_qc_map
from pprint import pprint
pprint(meas_qc.build_qc_map())

mask = meas_qc.get_unflagged_mask_for_multi_meas([0,1,2,3,4])
meas_analyze = LsLabAnalyze("/home/elias/proj/_dls_sql/markus_hfip_water_salt.lslab2", mask=mask)
print(meas_analyze.get_meas_reps())
print(meas_analyze.get_all_q())
print(meas_analyze.get_avg_gamma_per_meas())
print(meas_analyze.get_particle_radius(viscosity=0.89e-3))




