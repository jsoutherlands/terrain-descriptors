import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd

class TerrainAnalyzer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.terrains = []
        self.metrics = []
        self.data = {}  # {terrain: {metric: array}}
        # All outputs (images, csvs) will be saved to this folder
        self.output_dir = os.path.join(self.base_path, 'analysis_outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Load all metric TXT files under each terrain folder."""
        print("📂 Loading data...")

        # Find terrains: only folders that contain at least one .txt metric file
        candidates = [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]
        terrains = []
        for d in sorted(candidates):
            full = os.path.join(self.base_path, d)
            if any(f.endswith('.txt') for f in os.listdir(full)):
                terrains.append(d)
        self.terrains = terrains

        if not self.terrains:
            print("❌ No terrain subfolders found")
            return False

        # Pick first terrain that has metrics to discover metric names
        if len(self.terrains) > 0:
            first_terrain = os.path.join(self.base_path, self.terrains[0])
            metric_files = [f for f in os.listdir(first_terrain) if f.endswith('.txt')]
            self.metrics = sorted([os.path.splitext(f)[0] for f in metric_files])
        else:
            self.metrics = []

        print(f"✅ Found {len(self.terrains)} terrains: {self.terrains}")
        print(f"✅ Found {len(self.metrics)} metrics: {self.metrics}")

        # Load data
        for terrain in self.terrains:
            self.data[terrain] = {}
            for metric in self.metrics:
                file_path = os.path.join(self.base_path, terrain, f"{metric}.txt")
                if not os.path.exists(file_path):
                    print(f"   ⚠️ Missing file: {file_path}")
                    continue

                try:
                    arr = np.loadtxt(file_path).astype(float).flatten()
                except Exception as e:
                    print(f"   ❌ Error reading {file_path}: {e}")
                    continue

                finite_mask = np.isfinite(arr)
                if not np.all(finite_mask):
                    n_bad = np.count_nonzero(~finite_mask)
                    print(f"   ⚠️ {file_path}: found {n_bad} non-finite values; they will be removed")
                    arr = arr[finite_mask]

                if arr.size == 0:
                    print(f"   ⚠️ {file_path}: no valid data after filtering; skipping")
                    continue

                self.data[terrain][metric] = arr
                print(f"   📊 {terrain}/{metric}: loaded {arr.size:,} points")

        return True
    
    def plot_individual_histograms(self, num_bins=25):
        """Generate and save a multi-panel histogram image for each terrain.

        Saves: <output_dir>/<terrain>_individual_histograms.png
        """
        print("\n📊 Generating individual histograms (saved to output folder)...")

        for terrain in self.terrains:
            metrics_present = [m for m in self.metrics if m in self.data.get(terrain, {})]
            n_metrics = len(metrics_present)
            if n_metrics == 0:
                continue

            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_metrics == 1:
                axes = [axes]
            elif n_rows > 1 and n_cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes] if n_cols == 1 else axes

            axes_used = 0
            for metric in metrics_present:
                data_arr = self.data[terrain][metric]
                if data_arr.size == 0:
                    continue

                try:
                    mean = np.mean(data_arr)
                    std = np.std(data_arr)
                except Exception:
                    continue
                # Compute histogram (we compute explicitly to get bin width and probabilities)
                hist_counts, edges = np.histogram(data_arr, bins=num_bins)
                n_pts = data_arr.size
                bin_width = edges[1] - edges[0]
                # probability per bin (sums to 1)
                bin_prob = hist_counts.astype(float) / hist_counts.sum() if hist_counts.sum() > 0 else np.zeros_like(hist_counts, dtype=float)
                # density = prob / bin_width (what matplotlib shows when density=True)
                density = bin_prob / bin_width

                axes[axes_used].hist(data_arr, bins=edges, alpha=0.7,
                                     color='skyblue', edgecolor='black', density=True)
                axes[axes_used].axvline(mean, color='red', linestyle='--', label=f'mean={mean:.4g}')
                axes[axes_used].axvline(mean + std, color='orange', linestyle=':', alpha=0.7, label='mean±std')
                axes[axes_used].axvline(mean - std, color='orange', linestyle=':', alpha=0.7)

                # Annotate top bin as percent of total
                max_idx = int(np.argmax(bin_prob)) if bin_prob.size > 0 else 0
                max_prob = float(bin_prob[max_idx]) if bin_prob.size > 0 else 0.0
                max_left, max_right = edges[max_idx], edges[max_idx+1]
                percent = max_prob * 100.0
                axes[axes_used].annotate(f'{percent:.2f}% in [{max_left:.4g},{max_right:.4g}]',
                                         xy=((max_left + max_right) / 2, density[max_idx]),
                                         xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8,
                                         arrowprops=dict(arrowstyle='->', lw=0.5))

                # Overlay a sampled KDE for visual smoothing (sample up to 100k points for performance)
                try:
                    sample_size = 100000
                    if n_pts > sample_size:
                        sample = np.random.choice(data_arr, sample_size, replace=False)
                    else:
                        sample = data_arr
                    kde = stats.gaussian_kde(sample)
                    xs = np.linspace(edges[0], edges[-1], 1024)
                    kde_vals = kde(xs)
                    axes[axes_used].plot(xs, kde_vals, color='navy', linewidth=1.0, label='KDE')
                except Exception:
                    # If KDE fails, continue without it
                    pass

                axes[axes_used].set_title(f"{metric}  (n={n_pts:,}, bin_width={bin_width:.4g})")
                axes[axes_used].set_xlabel('Value')
                axes[axes_used].set_ylabel('Density')
                axes[axes_used].legend()
                axes[axes_used].grid(True, alpha=0.3)

                axes_used += 1

            # Hide unused axes
            for j in range(axes_used, len(axes)):
                axes[j].set_visible(False)

            plt.suptitle(f'Distributions - {terrain}', fontsize=16, y=0.98)
            plt.tight_layout()
            out_path = os.path.join(self.output_dir, f'{terrain}_individual_histograms.png')
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"   ✅ Saved: {out_path}")
    
    def compare_terrains_by_metric(self, num_bins=125, log_scale=False):
        """For each metric, create a comparative histogram across terrains and save it.

        Saves: <output_dir>/compare_<metric>.png
        """
        print(f"\n🔍 Comparing metrics between terrains (saved to output folder)...")

        for metric in self.metrics:
            print(f"   Processing: {metric}")
            terrains_with = [t for t in self.terrains if metric in self.data.get(t, {})]
            if len(terrains_with) < 2:
                print(f"   ❌ Not enough terrains with {metric}")
                continue

            all_vals = np.hstack([self.data[t][metric] for t in terrains_with])
            if all_vals.size == 0:
                print(f"   ⚠️ {metric}: no valid data, skipping")
                continue

            vmin, vmax = np.min(all_vals), np.max(all_vals)
            bins = np.linspace(vmin, vmax, num_bins)

            fig, ax = plt.subplots(figsize=(12, 6))
            # Plot each terrain and add sample size to label; overlay KDE (sampled)
            for idx, terrain in enumerate(terrains_with):
                arr = self.data[terrain][metric]
                n_pts = arr.size
                label = f"{terrain} (n={n_pts:,})"
                ax.hist(arr, bins=bins, alpha=0.45, label=label, density=True,
                        histtype='stepfilled', edgecolor='black', linewidth=0.5)
                # KDE overlay (sample for performance)
                try:
                    sample_size = 100000
                    if n_pts > sample_size:
                        sample = np.random.choice(arr, sample_size, replace=False)
                    else:
                        sample = arr
                    kde = stats.gaussian_kde(sample)
                    xs = np.linspace(bins[0], bins[-1], 1024)
                    kde_vals = kde(xs)
                    ax.plot(xs, kde_vals, linewidth=1.0)
                except Exception:
                    pass

            ax.set_title(f'Comparison of {metric} across terrains')
            ax.set_xlabel(metric)
            ax.set_ylabel('Probability density')
            if log_scale:
                ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)

            out_path = os.path.join(self.output_dir, f'compare_{metric}.png')
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"   ✅ Saved: {out_path}")
    
    def compute_wasserstein_all(self):
        """Compute Wasserstein distances between terrains for each metric.

        Returns a dict: {metric: {'terrains': [...], 'distance_matrix': np.array}}
        """
        print("\n📏 Computing Wasserstein distances...")
        results = {}

        for metric in self.metrics:
            print(f"   Metric: {metric}")
            terrains_with = [t for t in self.terrains if metric in self.data.get(t, {})]
            if len(terrains_with) < 2:
                continue

            n = len(terrains_with)
            dist_matrix = np.full((n, n), np.nan)

            for i, t1 in enumerate(terrains_with):
                for j, t2 in enumerate(terrains_with):
                    if i <= j:
                        d1 = self.data[t1][metric]
                        d2 = self.data[t2][metric]
                        if d1.size == 0 or d2.size == 0:
                            continue
                        try:
                            d = wasserstein_distance(d1, d2)
                        except Exception as e:
                            print(f"      ❌ Error calculating Wasserstein {t1}↔{t2}: {e}")
                            d = np.nan
                        dist_matrix[i, j] = d
                        dist_matrix[j, i] = d
                        if i != j and np.isfinite(d):
                            print(f"      {t1:15} ↔ {t2:15}: {d:.6f}")

            results[metric] = {'terrains': terrains_with, 'distance_matrix': dist_matrix}

        return results
    
    def show_wasserstein_matrices(self, results):
        """Save Wasserstein distance matrices as images to the output folder."""
        print("\n📊 Saving Wasserstein matrices to output folder...")
        for metric, res in results.items():
            terrains = res['terrains']
            matrix = res['distance_matrix']
            if matrix.size == 0:
                continue

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(matrix, cmap='YlOrRd', interpolation='nearest')
            ax.set_title(f'Wasserstein - {metric}')
            ax.set_xticks(range(len(terrains)))
            ax.set_yticks(range(len(terrains)))
            ax.set_xticklabels(terrains, rotation=45, ha='right')
            ax.set_yticklabels(terrains)

            # Add numeric values
            max_val = np.nanmax(matrix)
            for i in range(len(terrains)):
                for j in range(len(terrains)):
                    val = matrix[i, j]
                    s = 'nan' if np.isnan(val) else f'{val:.3f}'
                    color = 'white' if (not np.isnan(max_val) and val > max_val / 2) else 'black'
                    ax.text(j, i, s, ha='center', va='center', color=color, fontsize=9, fontweight='bold')

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            out_path = os.path.join(self.output_dir, f'wasserstein_{metric}.png')
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"   ✅ Saved: {out_path}")
    
    def analyze_wasserstein_patterns(self, results):
        """Detect patterns and save dendrograms/reports to output folder."""
        print("\n🔍 Analyzing Wasserstein patterns...")
        for metric, res in results.items():
            print(f"\n   📈 PATTERNS IN {metric.upper()}:" )
            matrix = res['distance_matrix']
            terrains = res['terrains']
            if len(terrains) < 2:
                continue

            mask = ~np.eye(len(terrains), dtype=bool)
            off_diag = matrix[mask]
            if off_diag.size > 0:
                min_val = np.nanmin(off_diag)
                max_val = np.nanmax(off_diag)
                idx_min = np.where((matrix == min_val) & mask)
                if len(idx_min[0]) > 0:
                    i_min, j_min = idx_min[0][0], idx_min[1][0]
                    print(f"      Most similar pair: {terrains[i_min]} ↔ {terrains[j_min]} ({min_val:.4f})")
                idx_max = np.where(matrix == max_val)
                if len(idx_max[0]) > 0:
                    i_max, j_max = idx_max[0][0], idx_max[1][0]
                    if i_max != j_max:
                        print(f"      Most different pair: {terrains[i_max]} ↔ {terrains[j_max]} ({max_val:.4f})")

            if len(terrains) > 2:
                try:
                    Z = linkage(matrix, method='average')
                    fig = plt.figure(figsize=(10, 4))
                    dendrogram(Z, labels=terrains, leaf_rotation=45)
                    plt.title(f'Clustering by Similarity - {metric}')
                    plt.ylabel('Wasserstein distance')
                    out_path = os.path.join(self.output_dir, f'dendrogram_{metric}.png')
                    plt.tight_layout()
                    plt.savefig(out_path, dpi=150)
                    plt.close(fig)
                    print(f"   ✅ Saved dendrogram: {out_path}")
                except Exception as e:
                    print(f"      ❌ Clustering error: {e}")
    
    def find_best_synthetic(self, results, reference_terrain="real"):
        """Find the synthetic terrain that is most similar to the reference terrain.

        Returns the best match dict or None.
        """
        print(f"\n🏆 Finding best synthetic terrain (reference: {reference_terrain})")
        best = []
        for metric, res in results.items():
            terrains = res['terrains']
            matrix = res['distance_matrix']
            if reference_terrain in terrains:
                idx_ref = terrains.index(reference_terrain)
                for j, t in enumerate(terrains):
                    if t != reference_terrain:
                        best.append({'metric': metric, 'synthetic_terrain': t, 'distance': matrix[idx_ref, j]})

        if not best:
            print("   ❌ No synthetic terrains found to compare")
            return None

        best.sort(key=lambda x: x['distance'])
        print("   Ranking of synthetic terrains (most realistic first):")
        for i, r in enumerate(best[:10]):
            print(f"      {i+1:2d}. {r['synthetic_terrain']:15} ({r['metric']:15}): {r['distance']:.4f}")
        return best[0]
    
    def identify_critical_metrics(self, results, reference_terrain="real"):
        """Identify metrics that best discriminate between reference and synthetics.

        Returns list of (metric, mean_distance) sorted desc.
        """
        print(f"\n🎯 Identifying most discriminative metrics")
        discr = {}
        for metric, res in results.items():
            terrains = res['terrains']
            matrix = res['distance_matrix']
            if reference_terrain in terrains:
                idx_ref = terrains.index(reference_terrain)
                dists = [matrix[idx_ref, j] for j, t in enumerate(terrains) if t != reference_terrain]
                if dists:
                    discr[metric] = np.nanmean(dists)

        if not discr:
            print("   ❌ Could not compute discrimination")
            return []

        ordered = sorted(discr.items(), key=lambda x: x[1], reverse=True)
        print("   Most discriminative metrics (highest to lowest):")
        for i, (m, v) in enumerate(ordered):
            print(f"      {i+1:2d}. {m:15}: {v:.4f}")
        return ordered
    
    def generate_statistical_report(self):
        """Generate and return a DataFrame with summary statistics for each terrain/metric.

        Also saves the report CSV to the output folder.
        """
        print("\n📈 Generating statistical report...")
        rows = []
        for terrain in self.terrains:
            print(f"\nTerrain: {terrain}")
            for metric in self.metrics:
                if metric in self.data.get(terrain, {}):
                    arr = np.asarray(self.data[terrain][metric])
                    if arr.size == 0:
                        continue
                    try:
                        mean = float(np.mean(arr))
                        median = float(np.median(arr))
                        std = float(np.std(arr))
                        vmin = float(np.min(arr))
                        vmax = float(np.max(arr))
                    except Exception:
                        mean = median = std = vmin = vmax = float('nan')

                    try:
                        skewness = float(stats.skew(arr))
                    except Exception:
                        skewness = float('nan')
                    try:
                        kurt = float(stats.kurtosis(arr))
                    except Exception:
                        kurt = float('nan')

                    print(f"   {metric:15}: mean={mean:.6g} | std={std:.6g} | min={vmin:.6g} | max={vmax:.6g} | skew={skewness:.6g}")
                    rows.append({
                        'terrain': terrain,
                        'metric': metric,
                        'mean': mean,
                        'median': median,
                        'std': std,
                        'min': vmin,
                        'max': vmax,
                        'skew': skewness,
                        'kurtosis': kurt,
                        'num_points': int(arr.size)
                    })

        self.df_statistics = pd.DataFrame(rows)
        out_csv = os.path.join(self.output_dir, 'statistical_report.csv')
        self.df_statistics.to_csv(out_csv, index=False)
        print(f"   ✅ Saved statistical report: {out_csv}")
        return self.df_statistics

    def export_histogram_features(self, n_bins=32, out_filename="features_histograms.csv", global_bins=True):
        """Export normalized histogram vectors and summary stats for ML.

        Columns: terrain, metric, bin_0 ... bin_{n_bins-1}, mean, std, skew, kurtosis, num_points
        Saves CSV to the output folder.
        """
        print(f"\n💾 Exporting histogram features to {out_filename} (n_bins={n_bins})...")
        rows = []

        # Precompute global bins per metric
        bins_by_metric = {}
        for metric in self.metrics:
            vals = []
            for terrain in self.terrains:
                if metric in self.data.get(terrain, {}):
                    vals.extend(self.data[terrain][metric])
            if len(vals) == 0:
                continue
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5
            bins_by_metric[metric] = np.linspace(vmin, vmax, n_bins + 1)

        for terrain in self.terrains:
            for metric in self.metrics:
                if metric not in self.data.get(terrain, {}):
                    continue
                arr = np.asarray(self.data[terrain][metric], dtype=float)
                npts = arr.size
                if npts == 0:
                    continue

                if global_bins and metric in bins_by_metric:
                    bins = bins_by_metric[metric]
                else:
                    vmin, vmax = float(np.min(arr)), float(np.max(arr))
                    if vmin == vmax:
                        vmin -= 0.5
                        vmax += 0.5
                    bins = np.linspace(vmin, vmax, n_bins + 1)

                hist, _ = np.histogram(arr, bins=bins)
                hist = hist.astype(float)
                s = hist.sum()
                if s > 0:
                    hist = hist / s
                else:
                    hist = np.zeros_like(hist, dtype=float)

                mean = float(np.mean(arr)) if npts > 0 else float('nan')
                std = float(np.std(arr)) if npts > 0 else float('nan')
                skew = float(stats.skew(arr)) if npts > 0 else float('nan')
                kurt = float(stats.kurtosis(arr)) if npts > 0 else float('nan')

                median = float(np.median(arr)) if npts > 0 else float('nan')
                vmin = float(np.min(arr)) if npts > 0 else float('nan')
                vmax = float(np.max(arr)) if npts > 0 else float('nan')
                row = {'terrain': terrain, 'metric': metric, 'mean': mean, 'median': median, 'std': std, 'min': vmin, 'max': vmax, 'skew': skew, 'kurtosis': kurt, 'num_points': int(npts)}
                for i in range(len(hist)):
                    row[f'bin_{i}'] = float(hist[i])
                rows.append(row)

        if len(rows) == 0:
            print("   ❌ No features to export")
            return None

        df = pd.DataFrame(rows)
        bin_cols = sorted([c for c in df.columns if c.startswith('bin_')], key=lambda x: int(x.split('_')[1]))
        cols = ['terrain', 'metric'] + bin_cols + ['mean', 'median', 'std', 'min', 'max', 'skew', 'kurtosis', 'num_points']
        df = df[cols]
        out_path = os.path.join(self.output_dir, out_filename)
        df.to_csv(out_path, index=False)
        print(f"   ✅ Saved features CSV: {out_path}")
        return out_path
    
    def save_results_csv(self, filename="resultados_terrenos.csv"):
        """Save the statistical report CSV into the output folder."""
        if hasattr(self, 'df_statistics'):
            out_path = os.path.join(self.output_dir, filename)
            self.df_statistics.to_csv(out_path, index=False)
            print(f"\n💾 Saved results CSV to {out_path}")
        else:
            print("\n❌ No statistical data to save")
    
    def run_full_analysis(self, reference_terrain="real"):
        print("🚀 STARTING FULL TERRAIN ANALYSIS")
        print("=" * 50)

        if not self.load_data():
            return

        # Individual histograms per terrain
        self.plot_individual_histograms()

        # Comparative histograms for each metric
        self.compare_terrains_by_metric()

        # Wasserstein analysis
        w_results = self.compute_wasserstein_all()
        if w_results:
            self.show_wasserstein_matrices(w_results)
            self.analyze_wasserstein_patterns(w_results)
            best = self.find_best_synthetic(w_results, reference_terrain)
            critical = self.identify_critical_metrics(w_results, reference_terrain)

        # Statistical report
        self.generate_statistical_report()

        # Export histogram features
        try:
            self.export_histogram_features(n_bins=32, out_filename="features_histograms.csv", global_bins=True)
        except Exception as e:
            print(f"   ❌ Error exporting histogram features: {e}")

        print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY")

# 🎯 USO PRÁCTICO - EJECUCIÓN INMEDIATA
if __name__ == "__main__":
    BASE_PATH = r"c:\Users\santi\terrain-descriptors-1\terrains\resultsPipeline"
    analyzer = TerrainAnalyzer(BASE_PATH)
    analyzer.run_full_analysis(reference_terrain="output_image_45.0_45.5_1.0_1.5")