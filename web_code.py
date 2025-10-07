import os
os.environ["PYNWB_NO_CACHE_DIR"] = "1"          #disable typemap caching
os.environ.setdefault("PYNWB_CACHE_DIR", "/tmp/pynwb_cache")  #just in case

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import tempfile
from sklearn.neighbors import KNeighborsClassifier
import torch.nn.functional as F
from bokeh.models import ColumnDataSource
import torch
import tarfile
from neurocurator import Neurocurator

from utils import normalize_to_minus1_1, normalize_by_row_max, plotter, compute_umap, acqm_file_reader, csv_downloader, compute_pumap, HIPPIE, compue_the_clusters, load_data_classifier


st.set_page_config(layout="wide", page_title="Neural data visualizer", page_icon=":bar_chart:")
st.title("Neural data visualizer")
st.write("Upload your CSV data files and visualize them please")



#load the curated dataset
acg_a, isi_a, wf_a, ct_a = load_data_classifier('classifying_data.tar.xz')

with st.container():
    uploaded_file_cell_type = st.file_uploader("upload the cell_type .csv file if you have one")
    if uploaded_file_cell_type is not None:

        # Can be used wherever a "file-like" object is accepted:
        dataframe_cell_type = pd.read_csv(uploaded_file_cell_type).astype(str)
        #st.write(dataframe_isi_dist)

        #dataframe_cell_type.iloc[:, 0] = dataframe_cell_type.iloc[:, 0].apply(lambda x: x.decode() if isinstance(x, bytes) else x)


uploaded_acg_files = []
uploaded_isi_files = []
uploaded_waveform_files = []
uploaded_phisiological_data_zip = []
nwb_uploaded = []
phy_uploaded = []
token_csv = False
token_acqm = False
token_nwb = False
token_phy = False

uploading_option = st.radio(
    "Choose input method:",
    ("Work with csv files", "Work with acqm.zip files", "Work with nwb files", "Work with phy files")
)

if uploading_option == "Work with csv files":
    col1, col2, col3 = st.columns(3)
    with col1:
        uploaded_acg_files = st.file_uploader(
            "Upload the acg .csv files here", accept_multiple_files=True, type=["csv"]
        ) or []
    with col2:
        uploaded_isi_files = st.file_uploader(
            "Upload the isi .csv files here", accept_multiple_files=True, type=["csv"]
        ) or []
    with col3:
        uploaded_waveform_files = st.file_uploader(
            "Upload the waveform .csv files here", accept_multiple_files=True, type=["csv"]
        ) or []

    # tokens become True only when lists are non-empty
    if len(uploaded_acg_files) > 0 and len(uploaded_isi_files) > 0 and len(uploaded_waveform_files) > 0:
        token_csv = True

    # make sure the other inputs are clearly "unused"
    uploaded_phisiological_data_zip = []
    nwb_uploaded = []

elif uploading_option == "Work with acqm.zip files":
    uploaded_phisiological_data_zip = st.file_uploader(
        "Upload your acqm.zip files here", accept_multiple_files=True, type=["zip"]
    ) or []

    if len(uploaded_phisiological_data_zip) > 0:
        token_acqm = True

    uploaded_acg_files = []
    uploaded_isi_files = []
    uploaded_waveform_files = []
    nwb_uploaded = []

elif uploading_option == "Work with nwb files":
    nwb_uploaded = st.file_uploader(
        "Upload your nwb files here", type=["nwb"], accept_multiple_files=True
    ) or []

    if len(nwb_uploaded) > 0:
        token_nwb = True

    uploaded_phisiological_data_zip = []
    uploaded_acg_files = []
    uploaded_isi_files = []
    uploaded_waveform_files = []

elif uploading_option == "Work with phy files":
    phy_uploaded = st.file_uploader(
        "Upload your phy files here", type=["zip"], accept_multiple_files=True
    ) or []

    if len(phy_uploaded) > 0:
        token_phy = True

    uploaded_phisiological_data_zip = []
    uploaded_acg_files = []
    uploaded_isi_files = []
    uploaded_waveform_files = []


if token_acqm or token_csv or token_nwb or token_phy:


    if token_acqm == True:
        df_acg = pd.DataFrame()
        df_isi = pd.DataFrame()
        df_waveforms = pd.DataFrame()

        for uploaded_file in uploaded_phisiological_data_zip:

            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            uploaded_file_acg, uploaded_file_isi_dist, uploaded_file_waefroms = acqm_file_reader(tmp_file_path)

            df_acg = pd.concat([df_acg, uploaded_file_acg], ignore_index=True)
            df_isi = pd.concat([df_isi, uploaded_file_isi_dist], ignore_index=True)
            df_waveforms = pd.concat([df_waveforms, uploaded_file_waefroms], ignore_index=True)
    
    elif token_csv == True:
        df_acg = pd.DataFrame()
        df_isi = pd.DataFrame()
        df_waveforms = pd.DataFrame()

        for uploaded_file in uploaded_acg_files:
            df_acg = pd.concat([df_acg, pd.read_csv(uploaded_file)], ignore_index=True)

        for uploaded_file in uploaded_isi_files:
            df_isi = pd.concat([df_isi, pd.read_csv(uploaded_file)], ignore_index=True)

        for uploaded_file in uploaded_waveform_files:
            df_waveforms = pd.concat([df_waveforms, pd.read_csv(uploaded_file)], ignore_index=True)
    
    elif token_nwb == True :
        df_acg = pd.DataFrame()
        df_isi = pd.DataFrame()
        df_waveforms = pd.DataFrame()

        for uploaded_file in nwb_uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nwb') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            nc = Neurocurator()

            nc.load_nwb_spike_times(tmp_file_path)                            # → self.spike_times_train (en ms)
            nc.load_nwb_waveforms(tmp_file_path, n_datapoints=50,             # → self.waveforms (50 puntos)
            candidates=("waveform_mean", "spike_waveforms"))

            # Calcular ISI (0–100 ms) y ACG (±100 ms, bin=1 ms) con tus funciones ya existentes
            nc.isi_distribution = nc.compute_isi_distribution(time_window=100)
            nc.acgs = nc.compute_autocorrelogram(nc.spike_times_train)

            df_acg = pd.concat([df_acg, nc.acgs], ignore_index=True)
            df_isi = pd.concat([df_isi, nc.isi_distribution], ignore_index=True)
            df_waveforms = pd.concat([df_waveforms, nc.waveforms], ignore_index=True)


    
    elif token_phy == True:
        df_acg = pd.DataFrame()
        df_isi = pd.DataFrame()
        df_waveforms = pd.DataFrame()

        for uploaded_file in phy_uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            nc = Neurocurator()
            nc.load_phy_curated(tmp_file_path)
            
            df_acg = pd.concat([df_acg, nc.acgs], ignore_index=True)
            df_isi = pd.concat([df_isi, nc.isi_distribution], ignore_index=True)
            df_waveforms = pd.concat([df_waveforms, nc.waveforms], ignore_index=True)



#################################




###################################
    col1, col2, col3 = st.columns(3)
    with col1:
    #acg file
        resized_acg = F.interpolate(
            torch.tensor(df_acg.values, dtype=torch.float32).unsqueeze(1),
            size=100,
            mode='linear'
        ).squeeze(1).numpy()
        normalized_acg = normalize_to_minus1_1(df_acg)
        p = plotter(normalized_acg, 'ACG', 'Timepoint', 'Amplitude')
        st.bokeh_chart(p, use_container_width=True)

    with col2:
    #isi plot
        resized_isi = F.interpolate(
            torch.tensor(df_isi.values, dtype=torch.float32).unsqueeze(1),
            size=100,
            mode='linear'
        ).squeeze(1).numpy()
        normalized_isi = normalize_by_row_max(df_isi)
        p = plotter(normalized_isi, 'ISI distribution', 'Timepoint', 'Amplitude')
        st.bokeh_chart(p, use_container_width=True)
    
    with col3:
    #waveforms plot
        resized_waveforms = F.interpolate(
            torch.tensor(df_waveforms.values, dtype=torch.float32).unsqueeze(1),
            size=50,
            mode='linear'
        ).squeeze(1).numpy()
        normalized_waveforms = normalize_to_minus1_1(df_waveforms)
        p = plotter(normalized_waveforms, 'Waveforms', 'Timepoint', 'Amplitude')
        st.bokeh_chart(p, use_container_width=True)
    

        
    resized_acg_a = F.interpolate(
        torch.tensor(acg_a.values, dtype=torch.float32).unsqueeze(1),
        size=100,
        mode='linear'
    ).squeeze(1).numpy()
            
    resized_isi_a = F.interpolate(
                torch.tensor(isi_a.values, dtype=torch.float32).unsqueeze(1),
                size=100,
                mode='linear'
            ).squeeze(1).numpy()
        
    resized_wf_a = F.interpolate(
        torch.tensor(wf_a.values, dtype=torch.float32).unsqueeze(1),
        size=50,
        mode='linear'
    ).squeeze(1).numpy()

    #make dropdown panel to choose the source for HIPPIE model
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

    #get HIPPIE embedings

    acg_T = pd.concat([pd.DataFrame(resized_acg_a), pd.DataFrame(resized_acg)], ignore_index=True)
    isi_T = pd.concat([pd.DataFrame(resized_isi_a), pd.DataFrame(resized_isi)], ignore_index=True)
    wf_T = pd.concat([pd.DataFrame(resized_wf_a), pd.DataFrame(resized_waveforms)], ignore_index=True)


    #create a multimodal dataset with all modalities
    #also make it numpy arrays because MultiModalEphysDataset expects numPy arrays
    embedding, labels = HIPPIE(pd.DataFrame(acg_T), pd.DataFrame(isi_T), pd.DataFrame(wf_T), source)
    
    ##################################################
    #PUMAP
    #loading the onnx model
    output_array = compute_pumap(embedding)

    x = list(range(len(output_array)))
    y = output_array[:, 0]

    source = ColumnDataSource(data=dict(x=x, y=y))
    output_array = pd.DataFrame(output_array, columns=['UMAP 1', 'UMAP 2'])
    ################################################


    ##########################################################################
    #normal UMAP
    #embedding = compute_umap(embedding)
    #output_array = np.array(embedding, dtype=np.float32)
    #output_array = pd.DataFrame(output_array, columns=['UMAP 1', 'UMAP 2'])
    ##########################################################################

    if uploaded_file_cell_type is None:
        
        st.title('Parametric UMAP')

        #slider for the clusters
        num_neighbors = st.slider("Number of neighbors (Knn)", 2, 10, 5)

        #computing the kmeans clustering
        output_array = compue_the_clusters(output_array, num_neighbors, ct_a)

        #making the chart
        chart = alt.Chart(output_array).mark_circle(size=30).encode(
            x='UMAP 1',
            y='UMAP 2',
            color='Classifier:N',
        ).properties(
            width=800,
            height=800
        )
        st.altair_chart(chart, use_container_width=True)


        #here goes the choosing clusters thing
        option = st.selectbox(
            'Select a cluster to see the corresponding data points',
            options=output_array['Classifier'].unique()
        )
        st.write(f"You selected cluster {option}")

        selected = output_array[output_array['Classifier'] == option]
        acg_types = pd.concat([normalized_acg, output_array['Classifier']], axis=1)
        isi_types = pd.concat([normalized_isi, output_array['Classifier']], axis=1)
        waveforms_types = pd.concat([normalized_waveforms, output_array['Classifier']], axis=1)

        sel_acg_types = acg_types[acg_types['Classifier'] == option]
        sel_isi_types = isi_types[isi_types['Classifier'] == option]
        sel_waveforms_types = waveforms_types[waveforms_types['Classifier'] == option]


        ############################################

        #########Downloading files box##############
        #downloading_option = st.radio(
        #    "Choose a downloading option:",
        #    ("Download .csv (multiple files)", "Download .h5 (1 file)")
        #)
        #if downloading_option == "Download .h5 (1 file)":
        #    h5_downloader(output_array, acg_types, isi_types, waveforms_types)
        #elif downloading_option == "Download .csv (multiple files)":
        csv_downloader(output_array, acg_types, isi_types, waveforms_types)
            
        ############################################
        
        p = plotter(acg_types, 'ACG with clusters', 'Timepoint', 'Amplitude', option)
        st.bokeh_chart(p, use_container_width=True)
        
        p = plotter(isi_types, 'ISI distribution with clusters', 'Timepoint', 'Amplitude', option)
        st.bokeh_chart(p, use_container_width=True)
        
        p = plotter(waveforms_types, 'Waveforms with clusters', 'Timepoint', 'Amplitude', option)
        st.bokeh_chart(p, use_container_width=True)
    
    elif uploaded_file_cell_type is not None:
        
        st.title('Parametric UMAP with cell type information')
        output_array['Cluster'] = dataframe_cell_type

        #making the chart
        chart = alt.Chart(output_array).mark_circle(size=30).encode(
            x='UMAP 1',
            y='UMAP 2',
            color='Cluster',
        ).properties(
            width=800,
            height=800
        )
        st.altair_chart(chart, use_container_width=True)

        # Dropdown to select a type
        option = st.selectbox(
            'Select a cell type to see the corresponding data points',
            options=output_array['Cluster'].unique()
        )

        # Filter output_array and apply to each data modality
        selected = output_array[output_array['Cluster'] == option]
        acg_types = pd.concat([normalized_acg, output_array['Cluster']], axis=1)
        isi_types = pd.concat([normalized_isi, output_array['Cluster']], axis=1)
        waveforms_types = pd.concat([normalized_waveforms, output_array['Cluster']], axis=1)

        sel_acg_types = acg_types[acg_types['Cluster'] == option]
        sel_isi_types = isi_types[isi_types['Cluster'] == option]
        sel_waveforms_types = waveforms_types[waveforms_types['Cluster'] == option]

        # Download
        csv_downloader(output_array, acg_types, isi_types, waveforms_types)

        # Plotting
        p = plotter(acg_types, 'ACG with celltypes', 'Timepoint', 'Amplitude', option)
        st.bokeh_chart(p, use_container_width=True)

        p = plotter(isi_types, 'ISI with celltypes', 'Timepoint', 'Amplitude', option)
        st.bokeh_chart(p, use_container_width=True)

        p = plotter(waveforms_types, 'Waveforms with celltypes', 'Timepoint', 'Amplitude', option)
        st.bokeh_chart(p, use_container_width=True)

    

    #Ploting only with means per cluster 

    st.title('Ploting only with means per cluster')

    acg_mean_list = []
    isi_mean_list = []
    wf_mean_list = []
    cluster_labels = []

    if uploaded_file_cell_type is None:
        for label in acg_types['Classifier'].unique():
            acg_mean_list.append(np.mean(acg_types[acg_types['Classifier'] == label].drop(columns='Classifier'), axis=0))
            isi_mean_list.append(np.mean(isi_types[isi_types['Classifier'] == label].drop(columns='Classifier'), axis=0))
            wf_mean_list.append(np.mean(waveforms_types[waveforms_types['Classifier'] == label].drop(columns='Classifier'), axis=0))
            cluster_labels.append(label)


    elif uploaded_file_cell_type is not None:
        for x in output_array['Classifier'].unique():
            acg_mean_list.append(np.mean(acg_types[acg_types['Classifier']==x].drop(columns='Classifier'), axis=0))
            isi_mean_list.append(np.mean(isi_types[isi_types['Classifier']==x].drop(columns='Classifier'), axis=0))
            wf_mean_list.append(np.mean(waveforms_types[waveforms_types['Classifier']==x].drop(columns='Classifier'), axis=0))

            cluster_labels.append(x)
        
    acg_mean_list = pd.DataFrame(acg_mean_list)
    isi_mean_list = pd.DataFrame(isi_mean_list)
    wf_mean_list = pd.DataFrame(wf_mean_list)

    acg_mean_list['Classifier'] = cluster_labels
    isi_mean_list['Classifier'] = cluster_labels
    wf_mean_list['Classifier'] = cluster_labels



    p = plotter(pd.DataFrame(acg_mean_list), 'ACG_mean', 'Timepoint', 'Amplitude', selected_cluster=option, alpha_background=0.8, alpha_upfront=0.8, line_width_background=0.8, line_width_upfront=1)
    st.bokeh_chart(p, use_container_width=True)
    p=plotter(pd.DataFrame(isi_mean_list), 'isi_mean', 'Timepoint', 'Amplitude', selected_cluster=option, alpha_background=0.8, alpha_upfront=0.8, line_width_background=0.8, line_width_upfront=1)
    st.bokeh_chart(p, use_container_width=True)
    p=plotter(pd.DataFrame(wf_mean_list), 'Waveforms_mean', 'Timepoint', 'Amplitude', selected_cluster=option, alpha_background=0.8, alpha_upfront=0.8, line_width_background=0.8, line_width_upfront=1)
    st.bokeh_chart(p, use_container_width=True)

    

else:
    st.info("Please upload all required files (ACG, ISI distribution, and waveforms) to create the UMAP visualization.")