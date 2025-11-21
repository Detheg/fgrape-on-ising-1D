import sys, os, json, re
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from helpers import experiments_format
from scipy.optimize import curve_fit

# Default global plot style
# Sizes are in pt (1 pt = 1/72 inch = 0.3528 mm or 1 mm = 2.83465 pt)
""" List of all keys in plt.rcParams:
KeysView(RcParams({'_internal.classic_mode': False,
          'agg.path.chunksize': 0,
          'animation.bitrate': -1,
          'animation.codec': 'h264',
          'animation.convert_args': ['-layers', 'OptimizePlus'],
          'animation.convert_path': 'convert',
          'animation.embed_limit': 20.0,
          'animation.ffmpeg_args': [],
          'animation.ffmpeg_path': 'ffmpeg',
          'animation.frame_format': 'png',
          'animation.html': 'none',
          'animation.writer': 'ffmpeg',
          'axes.autolimit_mode': 'data',
          'axes.axisbelow': 'line',
          'axes.edgecolor': 'black',
          'axes.facecolor': 'white',
          'axes.formatter.limits': [-5, 6],
          'axes.formatter.min_exponent': 0,
          'axes.formatter.offset_threshold': 4,
          'axes.formatter.use_locale': False,
          'axes.formatter.use_mathtext': False,
          'axes.formatter.useoffset': True,
          'axes.grid': False,
          'axes.grid.axis': 'both',
          'axes.grid.which': 'major',
          'axes.labelcolor': 'black',
          'axes.labelpad': 4.0,
          'axes.labelsize': 'medium',
          'axes.labelweight': 'normal',
          'axes.linewidth': 0.8,
          'axes.prop_cycle': cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']),
          'axes.spines.bottom': True,
          'axes.spines.left': True,
          'axes.spines.right': True,
          'axes.spines.top': True,
          'axes.titlecolor': 'auto',
          'axes.titlelocation': 'center',
          'axes.titlepad': 6.0,
          'axes.titlesize': 'large',
          'axes.titleweight': 'normal',
          'axes.titley': None,
          'axes.unicode_minus': True,
          'axes.xmargin': 0.05,
          'axes.ymargin': 0.05,
          'axes.zmargin': 0.05,
          'axes3d.automargin': False,
          'axes3d.grid': True,
          'axes3d.mouserotationstyle': 'arcball',
          'axes3d.trackballborder': 0.2,
          'axes3d.trackballsize': 0.667,
          'axes3d.xaxis.panecolor': (0.95, 0.95, 0.95, 0.5),
          'axes3d.yaxis.panecolor': (0.9, 0.9, 0.9, 0.5),
          'axes3d.zaxis.panecolor': (0.925, 0.925, 0.925, 0.5),
          'backend': 'module://matplotlib_inline.backend_inline',
          'backend_fallback': True,
          'boxplot.bootstrap': None,
          'boxplot.boxprops.color': 'black',
          'boxplot.boxprops.linestyle': '-',
          'boxplot.boxprops.linewidth': 1.0,
          'boxplot.capprops.color': 'black',
          'boxplot.capprops.linestyle': '-',
          'boxplot.capprops.linewidth': 1.0,
          'boxplot.flierprops.color': 'black',
          'boxplot.flierprops.linestyle': 'none',
          'boxplot.flierprops.linewidth': 1.0,
          'boxplot.flierprops.marker': 'o',
          'boxplot.flierprops.markeredgecolor': 'black',
          'boxplot.flierprops.markeredgewidth': 1.0,
          'boxplot.flierprops.markerfacecolor': 'none',
          'boxplot.flierprops.markersize': 6.0,
          'boxplot.meanline': False,
          'boxplot.meanprops.color': 'C2',
          'boxplot.meanprops.linestyle': '--',
          'boxplot.meanprops.linewidth': 1.0,
          'boxplot.meanprops.marker': '^',
          'boxplot.meanprops.markeredgecolor': 'C2',
          'boxplot.meanprops.markerfacecolor': 'C2',
          'boxplot.meanprops.markersize': 6.0,
          'boxplot.medianprops.color': 'C1',
          'boxplot.medianprops.linestyle': '-',
          'boxplot.medianprops.linewidth': 1.0,
          'boxplot.notch': False,
          'boxplot.patchartist': False,
          'boxplot.showbox': True,
          'boxplot.showcaps': True,
          'boxplot.showfliers': True,
          'boxplot.showmeans': False,
          'boxplot.vertical': True,
          'boxplot.whiskerprops.color': 'black',
          'boxplot.whiskerprops.linestyle': '-',
          'boxplot.whiskerprops.linewidth': 1.0,
          'boxplot.whiskers': 1.5,
          'contour.algorithm': 'mpl2014',
          'contour.corner_mask': True,
          'contour.linewidth': None,
          'contour.negative_linestyle': 'dashed',
          'date.autoformatter.day': '%Y-%m-%d',
          'date.autoformatter.hour': '%m-%d %H',
          'date.autoformatter.microsecond': '%M:%S.%f',
          'date.autoformatter.minute': '%d %H:%M',
          'date.autoformatter.month': '%Y-%m',
          'date.autoformatter.second': '%H:%M:%S',
          'date.autoformatter.year': '%Y',
          'date.converter': 'auto',
          'date.epoch': '1970-01-01T00:00:00',
          'date.interval_multiples': True,
          'docstring.hardcopy': False,
          'errorbar.capsize': 0.0,
          'figure.autolayout': False,
          'figure.constrained_layout.h_pad': 0.04167,
          'figure.constrained_layout.hspace': 0.02,
          'figure.constrained_layout.use': False,
          'figure.constrained_layout.w_pad': 0.04167,
          'figure.constrained_layout.wspace': 0.02,
          'figure.dpi': 100.0,
          'figure.edgecolor': 'white',
          'figure.facecolor': 'white',
          'figure.figsize': [6.4, 4.8],
          'figure.frameon': True,
          'figure.hooks': [],
          'figure.labelsize': 'large',
          'figure.labelweight': 'normal',
          'figure.max_open_warning': 20,
          'figure.raise_window': True,
          'figure.subplot.bottom': 0.11,
          'figure.subplot.hspace': 0.2,
          'figure.subplot.left': 0.125,
          'figure.subplot.right': 0.9,
          'figure.subplot.top': 0.88,
          'figure.subplot.wspace': 0.2,
          'figure.titlesize': 'large',
          'figure.titleweight': 'normal',
          'font.cursive': ['Apple Chancery',
                           'Textile',
                           'Zapf Chancery',
                           'Sand',
                           'Script MT',
                           'Felipa',
                           'Comic Neue',
                           'Comic Sans MS',
                           'cursive'],
          'font.family': ['sans-serif'],
          'font.fantasy': ['Chicago',
                           'Charcoal',
                           'Impact',
                           'Western',
                           'xkcd script',
                           'fantasy'],
          'font.monospace': ['DejaVu Sans Mono',
                             'Bitstream Vera Sans Mono',
                             'Computer Modern Typewriter',
                             'Andale Mono',
                             'Nimbus Mono L',
                             'Courier New',
                             'Courier',
                             'Fixed',
                             'Terminal',
                             'monospace'],
          'font.sans-serif': ['DejaVu Sans',
                              'Bitstream Vera Sans',
                              'Computer Modern Sans Serif',
                              'Lucida Grande',
                              'Verdana',
                              'Geneva',
                              'Lucid',
                              'Arial',
                              'Helvetica',
                              'Avant Garde',
                              'sans-serif'],
          'font.serif': ['DejaVu Serif',
                         'Bitstream Vera Serif',
                         'Computer Modern Roman',
                         'New Century Schoolbook',
                         'Century Schoolbook L',
                         'Utopia',
                         'ITC Bookman',
                         'Bookman',
                         'Nimbus Roman No9 L',
                         'Times New Roman',
                         'Times',
                         'Palatino',
                         'Charter',
                         'serif'],
          'font.size': 10.0,
          'font.stretch': 'normal',
          'font.style': 'normal',
          'font.variant': 'normal',
          'font.weight': 'normal',
          'grid.alpha': 1.0,
          'grid.color': '#b0b0b0',
          'grid.linestyle': '-',
          'grid.linewidth': 0.8,
          'hatch.color': 'black',
          'hatch.linewidth': 1.0,
          'hist.bins': 10,
          'image.aspect': 'equal',
          'image.cmap': 'viridis',
          'image.composite_image': True,
          'image.interpolation': 'auto',
          'image.interpolation_stage': 'auto',
          'image.lut': 256,
          'image.origin': 'upper',
          'image.resample': True,
          'interactive': False,
          'keymap.back': ['left', 'c', 'backspace', 'MouseButton.BACK'],
          'keymap.copy': ['ctrl+c', 'cmd+c'],
          'keymap.forward': ['right', 'v', 'MouseButton.FORWARD'],
          'keymap.fullscreen': ['f', 'ctrl+f'],
          'keymap.grid': ['g'],
          'keymap.grid_minor': ['G'],
          'keymap.help': ['f1'],
          'keymap.home': ['h', 'r', 'home'],
          'keymap.pan': ['p'],
          'keymap.quit': ['ctrl+w', 'cmd+w', 'q'],
          'keymap.quit_all': [],
          'keymap.save': ['s', 'ctrl+s'],
          'keymap.xscale': ['k', 'L'],
          'keymap.yscale': ['l'],
          'keymap.zoom': ['o'],
          'legend.borderaxespad': 0.5,
          'legend.borderpad': 0.4,
          'legend.columnspacing': 2.0,
          'legend.edgecolor': '0.8',
          'legend.facecolor': 'inherit',
          'legend.fancybox': True,
          'legend.fontsize': 'medium',
          'legend.framealpha': 0.8,
          'legend.frameon': True,
          'legend.handleheight': 0.7,
          'legend.handlelength': 2.0,
          'legend.handletextpad': 0.8,
          'legend.labelcolor': 'None',
          'legend.labelspacing': 0.5,
          'legend.loc': 'best',
          'legend.markerscale': 1.0,
          'legend.numpoints': 1,
          'legend.scatterpoints': 1,
          'legend.shadow': False,
          'legend.title_fontsize': None,
          'lines.antialiased': True,
          'lines.color': 'C0',
          'lines.dash_capstyle': <CapStyle.butt: 'butt'>,
          'lines.dash_joinstyle': <JoinStyle.round: 'round'>,
          'lines.dashdot_pattern': [6.4, 1.6, 1.0, 1.6],
          'lines.dashed_pattern': [3.7, 1.6],
          'lines.dotted_pattern': [1.0, 1.65],
          'lines.linestyle': '-',
          'lines.linewidth': 1.5,
          'lines.marker': 'None',
          'lines.markeredgecolor': 'auto',
          'lines.markeredgewidth': 1.0,
          'lines.markerfacecolor': 'auto',
          'lines.markersize': 6.0,
          'lines.scale_dashes': True,
          'lines.solid_capstyle': <CapStyle.projecting: 'projecting'>,
          'lines.solid_joinstyle': <JoinStyle.round: 'round'>,
          'macosx.window_mode': 'system',
          'markers.fillstyle': 'full',
          'mathtext.bf': 'sans:bold',
          'mathtext.bfit': 'sans:italic:bold',
          'mathtext.cal': 'cursive',
          'mathtext.default': 'it',
          'mathtext.fallback': 'cm',
          'mathtext.fontset': 'dejavusans',
          'mathtext.it': 'sans:italic',
          'mathtext.rm': 'sans',
          'mathtext.sf': 'sans',
          'mathtext.tt': 'monospace',
          'patch.antialiased': True,
          'patch.edgecolor': 'black',
          'patch.facecolor': 'C0',
          'patch.force_edgecolor': False,
          'patch.linewidth': 1.0,
          'path.effects': [],
          'path.simplify': True,
          'path.simplify_threshold': 0.111111111111,
          'path.sketch': None,
          'path.snap': True,
          'pcolor.shading': 'auto',
          'pcolormesh.snap': True,
          'pdf.compression': 6,
          'pdf.fonttype': 3,
          'pdf.inheritcolor': False,
          'pdf.use14corefonts': False,
          'pgf.preamble': '',
          'pgf.rcfonts': True,
          'pgf.texsystem': 'xelatex',
          'polaraxes.grid': True,
          'ps.distiller.res': 6000,
          'ps.fonttype': 3,
          'ps.papersize': 'letter',
          'ps.useafm': False,
          'ps.usedistiller': None,
          'savefig.bbox': None,
          'savefig.directory': '~',
          'savefig.dpi': 'figure',
          'savefig.edgecolor': 'auto',
          'savefig.facecolor': 'auto',
          'savefig.format': 'png',
          'savefig.orientation': 'portrait',
          'savefig.pad_inches': 0.1,
          'savefig.transparent': False,
          'scatter.edgecolors': 'face',
          'scatter.marker': 'o',
          'svg.fonttype': 'path',
          'svg.hashsalt': None,
          'svg.id': None,
          'svg.image_inline': True,
          'text.antialiased': True,
          'text.color': 'black',
          'text.hinting': 'force_autohint',
          'text.hinting_factor': 8,
          'text.kerning_factor': 0,
          'text.latex.preamble': '',
          'text.parse_math': True,
          'text.usetex': False,
          'timezone': 'UTC',
          'tk.window_focus': False,
          'toolbar': 'toolbar2',
          'webagg.address': '127.0.0.1',
          'webagg.open_in_browser': True,
          'webagg.port': 8988,
          'webagg.port_retries': 50,
          'xaxis.labellocation': 'center',
          'xtick.alignment': 'center',
          'xtick.bottom': True,
          'xtick.color': 'black',
          'xtick.direction': 'out',
          'xtick.labelbottom': True,
          'xtick.labelcolor': 'inherit',
          'xtick.labelsize': 'medium',
          'xtick.labeltop': False,
          'xtick.major.bottom': True,
          'xtick.major.pad': 3.5,
          'xtick.major.size': 3.5,
          'xtick.major.top': True,
          'xtick.major.width': 0.8,
          'xtick.minor.bottom': True,
          'xtick.minor.ndivs': 'auto',
          'xtick.minor.pad': 3.4,
          'xtick.minor.size': 2.0,
          'xtick.minor.top': True,
          'xtick.minor.visible': False,
          'xtick.minor.width': 0.6,
          'xtick.top': False,
          'yaxis.labellocation': 'center',
          'ytick.alignment': 'center_baseline',
          'ytick.color': 'black',
          'ytick.direction': 'out',
          'ytick.labelcolor': 'inherit',
          'ytick.labelleft': True,
          'ytick.labelright': False,
          'ytick.labelsize': 'medium',
          'ytick.left': True,
          'ytick.major.left': True,
          'ytick.major.pad': 3.5,
          'ytick.major.right': True,
          'ytick.major.size': 3.5,
          'ytick.major.width': 0.8,
          'ytick.minor.left': True,
          'ytick.minor.ndivs': 'auto',
          'ytick.minor.pad': 3.4,
          'ytick.minor.right': True,
          'ytick.minor.size': 2.0,
          'ytick.minor.visible': False,
          'ytick.minor.width': 0.6,
          'ytick.right': False}))
"""
plt.rcParams.update({
    'font.size': 12,
    'xtick.labelsize' : 12,
    'ytick.labelsize' : 12,
    'axes.labelsize': 12,
    'legend.fontsize': 12,
    'legend.handlelength': 1.0, # shorter handles
    'legend.handletextpad': 0.5, # less padding between handle and text
    'legend.columnspacing': 1.0, # less spacing between columns
    'axes.linewidth': 0.800, # 0.282 mm
    'axes.formatter.use_mathtext': True,
    'xtick.major.width': 0.800, # 0.282 mm
    'xtick.minor.width': 0.800, # 0.211 mm
    'xtick.major.size': 2.551, # 0.9 mm
    'xtick.minor.size': 1.417, # 0.5 mm
    'xtick.minor.pad': 0.1,
    'xtick.direction': 'in',
    'xtick.minor.visible': True,
    'ytick.major.width': 0.800, # 0.282 mm
    'ytick.minor.width': 0.800, # 0.211 mm
    'ytick.major.size': 2.551, # 0.9 mm
    'ytick.minor.size': 1.417, # 0.5 mm
    'ytick.direction': 'in',
    'ytick.minor.visible': True,
    'lines.linewidth': 0.800, # 0.282 mm
    'figure.figsize': (6.1811, 4.8), # Default figsize in inches (6.1811 in = 157 mm is exactly page content width)
    'errorbar.capsize': 2.0, # a bit smaller cap size
    'axes.grid': False,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0,
    #'text.usetex': True, # Disable for faster plotting without LaTeX
    #'font.sans-serif' : 'Computer Modern Sans Serif', # Only works with LaTeX
})

from matplotlib.patheffects import Normal, SimpleLineShadow
white_outline = [SimpleLineShadow(shadow_color="white", linewidth=4, alpha=1, offset=(0,0)), Normal()] # White outline for better visibility

# Helper to open files
def open_from_dir(dir_, exclude=""):
    files = [f for f in os.listdir(dir_) if f.endswith(".json") or f.endswith(".npz")]
    data = []

    print("Opening files...")
    for file in tqdm(files):
        if exclude and re.search(exclude, file):
            continue
        with open(os.path.join(dir_, file), 'r') as f:
            # Extract fidelities
            if file.endswith(".json"):
                fidelities = json.load(f)["fidelity_each_timestep"]
            else:
                fidelities = np.load(os.path.join(dir_, file))["fidelities"]

            # Extract experiment parameters from filename
            params = {}
            for param, p_type, _, _, _ in experiments_format + [("s", int, None, None, None)]:
                match = re.search(rf"_{param}=((?:[A-Za-z]+|-?\d+(?:\.\d+)?)(?:,(?:[A-Za-z]+|-?\d+(?:\.\d+)?))*)", file) # Regex which selects numbers + letters or floating point numbers
                if match:
                    value_str = match.group(1)
                    if p_type == list:
                        # Convert string representation of list back to list
                        value = [int(x) for x in value_str]
                        params[param] = value
                    else:
                        params[param] = p_type(value_str)
                elif param != "s":
                    print(f"Parameter {param} not found in filename {file}.")

            data.append((
                file,
                params,
                np.array(fidelities)
            ))
    print("Done.")

    longest = max([len(d[2]) for d in data])
    fidelities_mat = np.full((len(data), longest), -1.0)  # Fill with -1.0 for missing values

    params_each = [d[1] for d in data]
    params_grouped = group_params(params_each)

    for i, d in enumerate(data):
        fidelities = d[2]
        fidelities_mat[i, :len(fidelities)] = fidelities

    return fidelities_mat, params_grouped, params_each

# Helper for data formatting
def group_params(params_each):
    params_grouped = {
        param: np.zeros(len(params_each), dtype=dtype) if dtype in [int, float] else [None]*len(params_each)
        for param, dtype, _, _, _ in experiments_format + [("s", int, None, None, None)]
    }
    for i,p in enumerate(params_each):
        for param in params_grouped.keys():
            if param in p:
                params_grouped[param][i] = p[param]

    return params_grouped

def select_best_runs(fidelities_mat, params_each, objective_fun=lambda fidelities: fidelities.mean(), group_by=["gammap", "gammam"]):
    # Group data by all but the sample "s" in filename. From each group, select the one with highest average fidelity
    best_data = {}
    for idx, (fidelities, group_key) in enumerate(zip(fidelities_mat, [tuple([params[group] for group in group_by]) for params in params_each])):
        value = objective_fun(fidelities[fidelities != -1.0]) # Exclude missing values (-1.0)
        if group_key not in best_data or value > best_data[group_key][1]:
            best_data[group_key] = (fidelities, value, idx)

    fidelities_best_mat = np.full((len(best_data), fidelities_mat.shape[1]), -1.0) # Fill with -1.0 for missing values
    params_grouped_best = {
        param: np.zeros(len(best_data), dtype=dtype) if dtype in [int, float] else [None]*len(best_data)
        for param, dtype, _, _, _ in experiments_format + [("s", int, None, None, None)]
    }
    params_each_best = []
    for i, (fidelities, _, idx) in enumerate(best_data.values()):
        fidelities_best_mat[i, :len(fidelities)] = fidelities
        for param in params_grouped_best.keys():
            if param in params_each[idx]:
                params_grouped_best[param][i] = params_each[idx][param]
        params_each_best.append(params_each[idx])

    return fidelities_best_mat, params_grouped_best, params_each_best

def grid_grouped_params(values, params_grouped, x_param, y_param):
    x_values = np.unique(params_grouped[x_param])
    y_values = np.unique(params_grouped[y_param])
    x_values.sort()
    y_values.sort()
    grid = np.zeros((len(y_values), len(x_values)))

    for i, (x, y) in enumerate(zip(params_grouped[x_param], params_grouped[y_param])):
        x_idx = np.where(x_values == x)[0][0]
        y_idx = np.where(y_values == y)[0][0]
        grid[y_idx, x_idx] = values[i]

    return x_values, y_values, grid

def sort_by(fidelities_mat, params_each, params_each2, ignored_keys=[], do_remove_missing=False):
    # Sort fidelities_mat and params_each to match entries in params_each2

    assert len(params_each) == len(params_each2) or do_remove_missing, "Length of params_each and params_each2 must be the same for sorting."
    idx = []
    for p2 in params_each2:
        for j, p in enumerate(params_each):
            match = True
            for key in p.keys():
                if key not in ignored_keys and p[key] != p2[key]:
                    match = False
                    break
            if match:
                idx.append(j)
                break
        else:
            if not do_remove_missing:
                raise ValueError(f"No matching entry found during sorting for p={p}.")

    fidelities_mat_sorted = fidelities_mat[idx, :]
    params_each_sorted = [params_each[i] for i in idx]

    return fidelities_mat_sorted, group_params(params_each_sorted), params_each_sorted
    
# Helper for analysis
def extract_time_constants(fidelities_mat):
    a_arr = np.zeros(fidelities_mat.shape[0])
    tau_arr = np.zeros(fidelities_mat.shape[0])
    a_std_arr = np.zeros(fidelities_mat.shape[0])
    tau_std_arr = np.zeros(fidelities_mat.shape[0])

    def exp_decay(t, a, tau):
        return a * np.exp(-t / tau) + (1 - a)

    for i, fidelities in enumerate(fidelities_mat):
        # Exclude missing values (-1.0) from fitting
        fidelities = fidelities[fidelities != -1.0]

        # Uncomment to fit exponential to region where fidelities close to their equilibrium value
        #equilibrium_value = np.mean(fidelities[-10:])
        #fidelity_cutoff = 1 - (1 - equilibrium_value) * 0.90  # 10% above equilibrium
        #idx = 0
        #for idx,f in enumerate(fidelities):
        #    if f < fidelity_cutoff: break
        #fidelities = fidelities[:idx]

        # Fit exponential decay to fidelities
        t_data = np.arange(len(fidelities))

        popt, pcov = curve_fit(exp_decay, t_data, fidelities, p0=(1, 10), bounds=(0, [1.0, np.inf]))
        a, tau = popt
        a_std, tau_std = np.sqrt(np.diag(pcov))

        a_arr[i] = a
        tau_arr[i] = tau
        a_std_arr[i] = a_std
        tau_std_arr[i] = tau_std

    return a_arr, tau_arr, a_std_arr, tau_std_arr

# Helper for plotting
def plot_runs(fidelities_mat, params_grouped, sort_by="s", show_labels=True, figsize=(10, 10)):
    # Sort by parameter
    labels = _labels_from_params(params_grouped, show_s=True)
    if sort_by:
        idx = np.argsort(params_grouped[sort_by])
        labels = [labels[i] for i in idx]
        fidelities_mat = fidelities_mat[idx, :]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        fidelities_mat, aspect='auto', origin='lower', 
        extent=[0, fidelities_mat.shape[1], 0, fidelities_mat.shape[0]],
        vmin=0, vmax=1,
        cmap='afmhot',
        interpolation=None,
    )
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Runs")
    if show_labels:
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, label="Fidelity")

    return fig, ax, im

def plot_runs_1D(fidelities_mat, params_grouped, figsize=(10, 10)):
    fig, ax = plt.subplots(1,1, figsize=figsize)

    gamma_p_min = np.min(params_grouped['gammap'])
    gamma_p_max = np.max(params_grouped['gammap'])
    gamma_m_min = np.min(params_grouped['gammam'])
    gamma_m_max = np.max(params_grouped['gammam'])

    for fidelity, gamma_p, gamma_m in zip(fidelities_mat, params_grouped['gammap'], params_grouped['gammam']):
        red = (np.log10(gamma_p) - np.log10(gamma_p_min)) / (np.log10(gamma_p_max) - np.log10(gamma_p_min))
        blue = (np.log10(gamma_m) - np.log10(gamma_m_min)) / (np.log10(gamma_m_max) - np.log10(gamma_m_min))
        color = [red, 0, blue]
        ax.plot(fidelity, alpha=0.5, color=color, label=f"$\\gamma_+={gamma_p}$, $\\gamma_-={gamma_m}$")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Normalized Fidelity")
    ax.grid()
    
    return fig, ax

def plot_grid(values, params_grouped, x_param="gammam", y_param="gammap", figsize=(4,3), vmin=0, vmax=1):
    x_values, y_values, grid = grid_grouped_params(values, params_grouped, x_param, y_param)

    fig, ax = plt.subplots(1,1, figsize=figsize)
    im = ax.pcolormesh(
        x_values,
        y_values,
        grid,
        cmap='afmhot',
        vmin = vmin,
        vmax = vmax,
    )
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)

    return fig, ax, im
    

def _labels_from_params(params_grouped, show_s=True):
    labels = []
    for gamma_p, gamma_m, s in zip(params_grouped['gammap'], params_grouped['gammam'], params_grouped['s']):
        if show_s:
            labels.append(f"$\\gamma_+={gamma_p}$, $\\gamma_-={gamma_m}$, $s={s}$")
        else:
            labels.append(f"$\\gamma_+={gamma_p}$, $\\gamma_-={gamma_m}$")
    return labels