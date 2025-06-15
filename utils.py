import numpy as np
import umap
from bokeh.plotting import figure
from bokeh.models import HoverTool
from neurocurator import Neurocurator
import streamlit as st
import io
import zipfile 
import pandas as pd 
import onnxruntime as ort
from sklearn.cluster import KMeans
import onnxruntime as ort
import streamlit as st



def get_embeddings_multimodal(loader, model):
    """Extract embeddings from a multimodal model."""
    model.eval()
    all_embeddings = []
    all_labels = []
    all_data = []
    
    with torch.no_grad():
        for sample in loader:
            embedding = model(sample)[0].detach().cpu().numpy()
            # Normalize embeddings
            embedding = (embedding - np.mean(embedding, axis=1, keepdims=True)) / np.std(embedding, axis=1, keepdims=True)
            all_embeddings.extend(embedding)
            label = sample[1]
            if label.ndim == 2:
                cls_label, source_label = label.unbind(1)
            else:
                cls_label = label
            all_labels.extend(cls_label.detach().cpu().numpy())
    
    return np.array(all_embeddings), np.array(all_labels)


##############
#My functions#
##############
#functions to normalize the data
@st.cache_data
def normalize_to_minus1_1(df):
    return df.apply(lambda row: 2 * ((row - row.min()) / (row.max() - row.min())) - 1, axis=1)

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
        cluster_col = data['Cluster']
        plot_data = data.drop('Cluster', axis=1)

        #Create masks for selected vs not selected
        selected_mask = cluster_col == selected_cluster

        #plot not selected clusters first so they go to the background
        if (~selected_mask).any():
            plot_lines(p, plot_data[~selected_mask], color="yellow", alpha=alpha_background, line_width=line_width_background)
        
        if selected_mask.any():
            plot_lines(p, plot_data[selected_mask], color="red", alpha=alpha_upfront, line_width=line_width_upfront)
    else:
        #0.3 1
        plot_lines(p, data, color="red", alpha=0.8, line_width=0.1)

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
    session = ort.InferenceSession("hippie_model_no_source.onnx")

    # Prepare inputs (no source required!)
    inputs = {
        "wave": wave.astype(np.float32),
        "isi": isi.astype(np.float32),
        "acg": acg.astype(np.float32),
    }

    # Run inference
    outputs = session.run(None, inputs)
    embedding = outputs[0]
    # For consistency, create dummy labels array if you still need labels
    labels = np.zeros((embedding.shape[0],), dtype=np.int64)

    return embedding, labels



@st.cache_data
def compute_umap(data):
    umap_model = umap.UMAP()
    embedding = umap_model.fit_transform(data)
    return embedding

@st.cache_resource
def compute_pumap(embedding):
    model = ort.InferenceSession("Mark_VII_model.onnx")
    input_name = model.get_inputs()[0].name

    # #TODO run model
    input = np.array(embedding, dtype=np.float32)
    output = model.run(None, {input_name: input})[0]

    output_array = np.array(output)

    return output_array

@st.cache_data
def compue_the_clusters(output_array, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    output_array['Cluster'] = kmeans.fit_predict(output_array[['UMAP 1', 'UMAP 2']])
    #embedding_df_formated = output_array.rename(columns={'Cluster': 'kmeans_types'})
    return output_array

@st.cache_data
def acqm_file_reader(tmp_file_path):
    
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