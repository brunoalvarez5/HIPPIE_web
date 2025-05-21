import streamlit as st
import pandas as pd
from io import StringIO
from bokeh.plotting import figure
from bokeh.models import HoverTool
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import altair as alt
import plotly.express as px
from streamlit_plotly_events import plotly_events
from bokeh.models import ColumnDataSource, CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import plotly.graph_objects as go
import onnxruntime as ort
import sys
import os
import tempfile
from neurocurator import Neurocurator
from sklearn.cluster import KMeans
import time
import hippie
from hippie.dataloading import MultiModalEphysDataset, EphysDatasetLabeled, BalancedBatchSampler, none_safe_collate


st.set_page_config(layout="wide", page_title="Neural data visualizer", page_icon=":bar_chart:")
st.title("Neural data visualizer")
st.write("Upload your CSV data files and visualize them please")

with st.container():
    uploaded_file_cell_type = st.file_uploader("upload the cell_type .csv file if you have one")
    if uploaded_file_cell_type is not None:
        # To read file as bytes:
        bytes_data_cell_type = uploaded_file_cell_type.getvalue()
        #st.write(bytes_data_isi_dist)

        # To convert to a string based IO:
        stringio_cell_type = StringIO(uploaded_file_cell_type.getvalue().decode("utf-8"))
        #st.write(stringio_isi_dist)

        # To read file as string:
        string_data_cell_type = stringio_cell_type.read()
        #st.write(string_data_isi_dist)

        # Can be used wherever a "file-like" object is accepted:
        dataframe_cell_type = pd.read_csv(uploaded_file_cell_type)
        #st.write(dataframe_isi_dist)

        dataframe_cell_type.iloc[:, 0] = dataframe_cell_type.iloc[:, 0].apply(lambda x: x.decode() if isinstance(x, bytes) else x)


uploaded_phisiological_data_zip = st.file_uploader("upload the acqm .zip file")

if uploaded_phisiological_data_zip is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        tmp_file.write(uploaded_phisiological_data_zip.read())
        tmp_file_path = tmp_file.name

    reader = Neurocurator()
    reader.load_acqm(tmp_file_path)

    uploaded_file_acg = reader.acgs
    uploaded_file_isi_dist = reader.isi_distribution
    uploaded_file_waefroms = reader.waveforms

    float64_cols_acg = uploaded_file_acg.columns
    float64_cols_isi = uploaded_file_isi_dist.columns
    float64_cols_waveforms = uploaded_file_waefroms.columns

    #uploaded_file_acg[float64_cols_acg] = uploaded_file_acg[float64_cols_acg].astype(np.float16)
    #uploaded_file_isi_dist[float64_cols_isi] = uploaded_file_isi_dist[float64_cols_isi].astype(np.float16)
    #uploaded_file_waefroms[float64_cols_waveforms] = uploaded_file_waefroms[float64_cols_waveforms].astype(np.float16)

    col1, col2, col3 = st.columns(3)

    #functions to normalize the data
    @st.cache_resource()
    def normalize_to_minus1_1(df):
        return df.apply(lambda row: 2 * ((row - row.min()) / (row.max() - row.min())) - 1, axis=1)
    @st.cache_resource()
    def normalize_by_row_max(df):
        return df.apply(lambda row: row / row.sum(), axis=0)
    
    @st.cache_resource()
    def plotter(data, title, x_label, y_label, selected_cluster=None):
        p = figure(
            title=title,
            x_axis_label=x_label,
            y_axis_label=y_label,
            width=800,
            height=600,
            tools='pan,wheel_zoom,box_zoom,reset'
        )
        p.background_fill_color = None
        p.border_fill_color = None
        p.xaxis.major_label_text_color = "white"
        p.yaxis.major_label_text_color = "white"
        p.xaxis.axis_label_text_color = "white"
        p.yaxis.axis_label_text_color = "white"
        p.title.text_color = "white"
        for index, row in data.iterrows():
            x = list(range(len(row)))
            y = row.values

            if selected_cluster is not None:
                line_color = "red" if row['Cluster'] == selected_cluster else "gray"
                line_alpha = 0.8 if row['Cluster'] == selected_cluster else 0.3
                line_width = 1 if row['Cluster'] == selected_cluster else 0.5
            else:  
                line_color = "red"
                line_alpha = 0.3
                line_width = 1

            p.line(x, y, line_width=line_width, alpha=line_alpha, color=line_color)
        #print("Yes I'm doing this again")
        p.add_tools(HoverTool(tooltips=[("x", "$x"), ("y", "$y")]))
        return p
    
    @st.cache_resource()
    def compute_umap(data):
        umap_model = umap.UMAP()
        embedding = umap_model.fit_transform(data)
        return embedding

    #acg file
    normalized_acg = normalize_to_minus1_1(uploaded_file_acg)
    p = plotter(normalized_acg, 'ACG', 'Timepoint', 'Amplitude')
    st.bokeh_chart(p)

    #isi plot
    normalized_isi = normalize_by_row_max(uploaded_file_isi_dist)
    p = plotter(normalized_isi, 'ISI distribution', 'Timepoint', 'Amplitude')
    st.bokeh_chart(p)
    
    #waveforms plot
    normalized_waveforms = normalize_to_minus1_1(uploaded_file_waefroms)
    p = plotter(normalized_waveforms, 'Waveforms', 'Timepoint', 'Amplitude')
    st.bokeh_chart(p)

    #TODO: pytorch dataloader for HIPPIE

    #load checkpoint
    #model = MultiModalCVAETrainModule.load_from_checkpoint("path/to/your/checkpoint.ckpt")
    #model.eval()
    
    #make dropdown panel with source
    dataset_files = {
        "braingeneers_manual_curation": 1,
        "cellexplorer_area": 2,
        "hausser": 3,
        "hull": 4,
        "lisberger": 5,
        "mouse_organoids_cell_line": 6
    }
    source = st.selectbox(
            'Select how your data was obtained',
            options=dataset_files.keys(),
        )
    source = dataset_files[source]

    #pass all normalized datasets to numpy
    normalized_acg_numpy = normalized_acg.to_numpy()
    normalized_isi_numpy = normalized_isi.to_numpy()
    normalized_waveforms_numpy = normalized_waveforms.to_numpy()

    #create a multimodal dataset with all modalities
    #TODO igualar dimensiones de todas a las default, wf 50, acg y isi 100 interpolate
    data_dict = {
        'wave': normalized_waveforms_numpy,
        'acg': normalized_acg_numpy,
        'isi': normalized_isi_numpy
    }

    dataset_multi = MultiModalEphysDataset(data_dict, source, mode='multi')

    #dataloader
    loader = DataLoader(
        dataset_multi,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    embedding, labels = get_embeddings_multimodal(loader, model)

    #TODO: get HIPPIE embedings
    #create a multimodal dataset with all modalities
    data_dict = {
        'wave': normalized_waveforms_numpy,
        'acg': normalized_acg_numpy,
        'isi': normalized_isi_numpy
    }

    dataset_multi = MultiModalEphysDataset(data_dict, source, mode='multi')

    #dataloader
    loader = DataLoader(
        dataset_multi,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    embedding, labels = get_embeddings_multimodal(loader, model)


    #TODO: Get UMAP embedings

    #UMAP
    #joining the datasets by rows
    combined_df = pd.concat([normalized_acg, normalized_waveforms, normalized_isi], axis=1)

    #dropping rows with NaN
    combined_df_clean = combined_df.dropna()
    st.write("Number of NaN rows dropped: ", len(combined_df)- len(combined_df_clean))
    
    embedding = compute_umap(combined_df_clean)

    embedding_df = pd.DataFrame(embedding, columns=['UMAP 1', 'UMAP 2'])

    if uploaded_file_cell_type is None:
        num_clusters = st.slider("Number of clusters (KMeans)", 2, 10, 5)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        embedding_df['Cluster'] = kmeans.fit_predict(embedding_df[['UMAP 1', 'UMAP 2']])
        embedding_df_formated = embedding_df.rename(columns={'Cluster': 'kmeans_types'})

        chart = alt.Chart(embedding_df).mark_circle(size=30).encode(
            x='UMAP 1',
            y='UMAP 2',
            color='Cluster:N',
        ).properties(
            width=800,
            height=800
        )
        st.altair_chart(chart, use_container_width=True)

        option = st.selectbox(
            'Select a cluster to see the corresponding data points',
            options=embedding_df_formated['kmeans_types'].unique()
        )
        st.write(f"You selected cluster {option}")

        selected = embedding_df[embedding_df['Cluster'] == option]
        acg_types = pd.concat([normalized_acg, embedding_df['Cluster']], axis=1)
        isi_types = pd.concat([normalized_isi, embedding_df['Cluster']], axis=1)
        waveforms_types = pd.concat([normalized_waveforms, embedding_df['Cluster']], axis=1)

        sel_acg_types = acg_types[acg_types['Cluster'] == option]
        sel_isi_types = isi_types[isi_types['Cluster'] == option]
        sel_waveforms_types = waveforms_types[waveforms_types['Cluster'] == option]

    
        p = plotter(acg_types, 'ACG_fancy', 'Timepoint', 'Amplitude', option)
        st.bokeh_chart(p)
    

        p = plotter(isi_types, 'ISI distribution_fancy', 'Timepoint', 'Amplitude', option)
        st.bokeh_chart(p)

        p = plotter(waveforms_types, 'Waveforms_fancy', 'Timepoint', 'Amplitude', option)
        st.bokeh_chart(p)



    elif uploaded_file_cell_type is not None:
        
        embedding_df_celltypes = pd.concat([embedding_df, dataframe_cell_type], axis=1)
        embedding_df_celltypes = pd.DataFrame(embedding_df_celltypes, columns=['UMAP 1', 'UMAP 2', 'types'])
        
        chart = alt.Chart(embedding_df).mark_circle(size=30).encode(
            x='UMAP 1',
            y='UMAP 2',
            color='0'
        ).properties(
            width=800,
            height=800
        )
        st.altair_chart(chart, use_container_width=True)


#embedding_df[types].isin(selected)

else:
    st.info("Please upload all required files (ACG, ISI distribution, and waveforms) to create the UMAP visualization.")