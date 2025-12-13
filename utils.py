import numpy as np
import umap
import streamlit as st
import io
import zipfile 
import pandas as pd 
import tarfile
import os


# def get_embeddings_multimodal(loader, model):
#     import torch
#     """Extract embeddings from a multimodal model."""
#     model.eval()
#     all_embeddings = []
#     all_labels = []
#     all_data = []
    
#     with torch.no_grad():
#         for sample in loader:
#             embedding = model(sample)[0].detach().cpu().numpy()
#             # Normalize embeddings
#             embedding = (embedding - np.mean(embedding, axis=1, keepdims=True)) / np.std(embedding, axis=1, keepdims=True)
#             all_embeddings.extend(embedding)
#             label = sample[1]
#             if label.ndim == 2:
#                 cls_label, source_label = label.unbind(1)
#             else:
#                 cls_label = label
#             all_labels.extend(cls_label.detach().cpu().numpy())
    
#     return np.array(all_embeddings), np.array(all_labels)


##############
#My functions#
##############
#functions to normalize the data
@st.cache_data
def normalize_to_minus1_1(df):
    import pandas as pd
    def norm_row(row):
        rmin = row.min()
        rmax = row.max()
        # If all values are identical -> avoid division by zero
        if rmax == rmin:
            # you can choose 0, or 1, or whatever; 0 is a neutral center
            return pd.Series(0.0, index=row.index)
        return 2 * ((row - rmin) / (rmax - rmin)) - 1

    return df.apply(norm_row, axis=1)

@st.cache_data
def normalize_by_row_max(df):
    return df.apply(lambda row: row / row.sum(), axis=0)


def plot_lines(ploted_obj, data, color, alpha, line_width):
    
    if data.empty:
        return

    values = data.values #convert to numpy for faster operations
    n_cols = values.shape[1]

    x_coords = list(range(n_cols))

    xs = [x_coords] * len(values) #whole empty list of lists defined
    ys = values.tolist()
    
    ploted_obj.multi_line(xs=xs, ys=ys, line_color=color, alpha=alpha, line_width=line_width)



@st.cache_resource
def plotter(data, title, x_label, y_label, selected_cluster=None, alpha_background=0.5, alpha_upfront=0.8, line_width_background=0.3, line_width_upfront=0.5):
    from bokeh.plotting import figure
    from bokeh.models import HoverTool
    p = figure(
        title=title,
        x_axis_label=x_label,
        y_axis_label=y_label,
        width=800,
        height=800,
        tools='pan,wheel_zoom,box_zoom,reset'
    )
    p.background_fill_color = None
    p.border_fill_color = None
    p.xaxis.major_label_text_color = "white"
    p.yaxis.major_label_text_color = "white"
    p.xaxis.axis_label_text_color = "white"
    p.yaxis.axis_label_text_color = "white"
    p.title.text_color = "white"
    p.add_tools(HoverTool(tooltips=[("x", "$x"), ("y", "$y")]))

    if selected_cluster is not None:

        #separate the column of the cluster
        cluster_col = data['Classifier']
        plot_data = data.drop('Classifier', axis=1)

        #Create masks for selected vs not selected
        selected_mask = cluster_col == selected_cluster

        #plot not selected clusters first so they go to the background
        if (~selected_mask).any():
            plot_lines(p, plot_data[~selected_mask], color="#FFB000", alpha=alpha_background, line_width=line_width_background)
        
        if selected_mask.any():
            plot_lines(p, plot_data[selected_mask], color="#00D8FF", alpha=alpha_upfront, line_width=line_width_upfront)
    else:
        #0.3 1
        plot_lines(p, data, color="#00D8FF", alpha=0.8, line_width=0.1)

    return p

@st.cache_resource
def load_model():
    #import model
    #create the base_model instance
    base_model = MultiModalCVAE(
        modalities={
            "wave": 50,
            "isi": 100,
            "acg": 100 #200 originaly, but we do interpolate to 100
        },
        z_dim=5,
        class_hidden_dim=5,
        num_sources=7,
        num_classes=5
    )

    # Load the full module with the base_model
    model = MultiModalCVAETrainModule.load_from_checkpoint(
        "/home/bruno/Documentos/GitHub/HIPPIE_web/epoch=49-step=950.ckpt",
        base_model=base_model,
        modality_weights={
            "wave": 1.0,
            "isi": 1.0,
            "acg": 1.0
        },
        learning_rate=1e-3,
        weight_decay=1e-5,
        beta=1.0
    )


    return model

def drop_nan_rows(*dfs):
    """Drop rows with NaNs from all dataframes simultaneously"""
    mask = ~(np.isnan(dfs[0]).any(axis=1) | np.isinf(dfs[0]).any(axis=1))
    for df in dfs[1:]:
        mask &= ~(np.isnan(df).any(axis=1) | np.isinf(df).any(axis=1))
    return [df[mask] for df in dfs]


@st.cache_resource
def HIPPIE(normalized_acg, normalized_isi, normalized_waveforms, source=None):
    import onnxruntime as ort
    """
    Compute the embedding using ONNX model (no source input required).
    """
    # Pass all normalized datasets to numpy
    normalized_acg_numpy = normalized_acg.values
    normalized_isi_numpy = normalized_isi.values
    normalized_waveforms_numpy = normalized_waveforms.values

    # Remove invalid rows (NaN, Inf)
    valid_mask = (
        ~np.isnan(normalized_acg_numpy).any(axis=1) & ~np.isinf(normalized_acg_numpy).any(axis=1) &
        ~np.isnan(normalized_isi_numpy).any(axis=1) & ~np.isinf(normalized_isi_numpy).any(axis=1) &
        ~np.isnan(normalized_waveforms_numpy).any(axis=1) & ~np.isinf(normalized_waveforms_numpy).any(axis=1)
    )
    normalized_acg_numpy = normalized_acg_numpy[valid_mask]
    normalized_isi_numpy = normalized_isi_numpy[valid_mask]
    normalized_waveforms_numpy = normalized_waveforms_numpy[valid_mask]

    # ONNX model expects [batch, 1, features]
    acg = normalized_acg_numpy[:, np.newaxis, :]
    isi = normalized_isi_numpy[:, np.newaxis, :]
    wave = normalized_waveforms_numpy[:, np.newaxis, :]

    # Load ONNX model (load once and cache)
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hippie_model_epoch=9-step=60.onnx")
    session = ort.InferenceSession(MODEL_PATH)


# ---- 4. Fix ACG length: model expects 200 ----
    expected_acg_len = 200
    current_acg_len = acg.shape[2]

    if current_acg_len != expected_acg_len:
        # Upsample along the last axis (simple linear interpolation)
        N = acg.shape[0]
        x_old = np.linspace(0.0, 1.0, current_acg_len)
        x_new = np.linspace(0.0, 1.0, expected_acg_len)
        acg_reshaped = np.empty((N, 1, expected_acg_len), dtype=np.float32)
        for i in range(N):
            # interpolate the 1D curve acg[i, 0, :]
            acg_reshaped[i, 0, :] = np.interp(x_new, x_old, acg[i, 0, :])
        acg = acg_reshaped

    batch_size = acg.shape[0]


    #make dummy labels
    source_labels = np.zeros((acg.shape[0],), dtype=np.int64)
    class_labels = np.zeros((acg.shape[0],), dtype=np.int64)

    # Prepare inputs (no source required!)
    inputs = {
        "wave": wave.astype(np.float32),
        "isi": isi.astype(np.float32),
        "acg": acg.astype(np.float32),
        "source_labels": source_labels,
        "class_labels": class_labels,
    }

    # Run inference
    outputs = session.run(None, inputs)
    embedding = outputs[0]
    # For consistency, create dummy labels array if you still need labels
    labels = np.zeros((embedding.shape[0],), dtype=np.int64)

    return embedding, labels



@st.cache_data
def compute_umap(data):
    """
    Compute a 2D UMAP embedding.

    If there are fewer than 2 points, UMAP can't build a graph and will crash,
    so we fall back to a trivial 2D projection using the first two dimensions
    of the input data.
    """
    data = np.asarray(data)

    # No points at all → return empty (caller should handle this case)
    if data.shape[0] == 0:
        return data

    # Only 1 point → UMAP cannot run; just take first 2 dims as a "fake UMAP"
    if data.shape[0] == 1:
        # data is (1, D). We want (1, 2)
        if data.shape[1] >= 2:
            return data[:, :2]
        else:
            # If D == 1, pad a second dimension with 0
            out = np.zeros((1, 2), dtype=float)
            out[0, 0] = data[0, 0]
            return out

    # Normal case: at least 2 points → run real UMAP
    umap_model = umap.UMAP()
    embedding = umap_model.fit_transform(data)
    return embedding


@st.cache_resource
def compute_pumap(embedding):
    import onnxruntime as ort
    model = ort.InferenceSession("Mark_VII_model.onnx")
    input_name = model.get_inputs()[0].name

    # #TODO run model
    input = np.array(embedding, dtype=np.float32)
    output = model.run(None, {input_name: input})[0]

    output_array = np.array(output)

    return output_array



@st.cache_data
def compue_the_clusters_labeled(output_array, num_neighbors, ct_a):
    from sklearn.neighbors import KNeighborsClassifier
    
    n = len(ct_a)
    X_labeled = output_array[['UMAP 1', 'UMAP 2']].iloc[:n].values
    y_labeled = ct_a.iloc[:, 0].values
    X_all = output_array[['UMAP 1', 'UMAP 2']].values

    # KNN classification for ALL points
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(X_labeled, y_labeled)
    output_array['Classifier'] = knn.predict(X_all)

    return output_array

    #embedding_df_formated = output_array.rename(columns={'Cluster': 'kmeans_types'})
    #return output_array

@st.cache_data
def compue_the_clusters_kmeans(output_array, num_neighbors):
    from sklearn.cluster import KMeans
    
    # Use UMAP coordinates of ALL points
    X_all = output_array[['UMAP 1', 'UMAP 2']].values

    # KMeans clustering for ALL points
    kmeans = KMeans(n_clusters=num_neighbors, n_init='auto', random_state=0)
    output_array['Classifier'] = kmeans.fit_predict(X_all)

    return output_array


@st.cache_data
def acqm_file_reader(tmp_file_path):
    from neurocurator import Neurocurator
    
    reader = Neurocurator()
    reader.load_acqm(tmp_file_path)

    uploaded_file_acg = reader.acgs
    uploaded_file_isi_dist = reader.isi_distribution
    uploaded_file_waefroms = reader.waveforms

    return uploaded_file_acg, uploaded_file_isi_dist, uploaded_file_waefroms

#@st.cache_data
def csv_downloader(embeddings, acg, isi, wf):

    dataframes = {
        "embeddings_clusters.csv": embeddings,
        "acg_clusters.csv": acg,
        "isi_clusters.csv": isi,
        "waveforms_clusters.csv": wf
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zipf:
        for filename, df in dataframes.items():
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            zipf.writestr(filename, csv_bytes)
    buffer.seek(0)
    
    st.download_button(
        label="Download all as zip",
        data=buffer.read(),
        file_name="datasets.zip",
        mime="application/zip"
    )

#@st.cache_data
def h5_downloader(embeddings, acg, isi, wf):
    import pandas as pd
    #create an in-memory bytes buffer
    buffer = io.BytesIO()

    #write DataFrames into HDF5 format
    with pd.HDFStore(buffer, mode='w') as store:
        store.put("embeddings_clusters", embeddings)
        store.put("acg_clusters", acg)
        store.put("isi_clusters", isi)
        store.put("waveforms_clusters", wf)

    buffer.seek(0)  #rewind buffer before sending to download

    #streamlit download button
    st.download_button(
        label="Download all as HDF5",
        data=buffer,
        file_name="datasets.h5",
        mime="application/octet-stream"
    )

#@st.cache_data
def load_data_classifier(tar_file_path):
    import pandas as pd
    
    csv_data = {}  # To store DataFrames

    with tarfile.open(tar_file_path, 'r:xz') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.csv'):
                # Read file object from tar
                f = tar.extractfile(member)
                # Load directly into pandas
                df = pd.read_csv(f)
                # Store in dictionary, key is the filename (you can change this)
                csv_data[member.name] = df

    return csv_data.get('acg.csv'), csv_data.get('isi_dist.csv'), csv_data.get('waveforms.csv'), csv_data.get('celltypes.csv')


@st.cache_data
def compue_the_clusters_hdbscan(output_array: pd.DataFrame, min_cluster_size: int, min_samples: int,) -> pd.DataFrame:
    """
    Cluster points in UMAP space using HDBSCAN.

    Parameters
    ----------
    output_array : pd.DataFrame
        Must contain columns 'UMAP 1' and 'UMAP 2'.
    min_cluster_size : int
        Smallest allowed cluster size. Higher -> fewer, more stable clusters.
    min_samples : int
        Density strictness. Higher -> only very dense regions form clusters.

    Returns
    -------
    pd.DataFrame
        Same as input but with:
            'Classifier'      : cluster labels (-1 = noise)
            'Cluster_prob'    : membership probability per point
    """
    import hdbscan
    #use UMAP coordinates of ALL points
    X_all = output_array[['UMAP 1', 'UMAP 2']].values

    #HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom',
        #metric='euclidean'  # default; uncomment if you want to be explicit
    )

    labels = clusterer.fit_predict(X_all)          #-1 = noise
    probs = clusterer.probabilities_              #membership strength [0,1]

    result = output_array.copy()
    result['Classifier'] = labels
    result['Cluster_prob'] = probs

    return result

def resize_rows_linear(arr: np.ndarray, out_len: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    n, in_len = arr.shape
    if in_len == out_len:
        return arr
    x_old = np.linspace(0.0, 1.0, in_len, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, out_len, dtype=np.float32)
    out = np.empty((n, out_len), dtype=np.float32)
    for i in range(n):
        out[i] = np.interp(x_new, x_old, arr[i]).astype(np.float32)
    return out
