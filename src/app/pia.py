"""
Privacy Impact Assessment (PIA) Analysis Application

This Streamlit app allows users to:
1. Load and explore PIA records from Excel files
2. Identify and categorize field types (free text, multichoice)
3. Filter records by Line of Business (LOB)
4. Perform text clustering on free text fields
5. Generate summaries of text clusters using open-source LLMs
6. Analyze records based on multichoice fields
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import utility modules
from src.utils import data_loader
# from src.utils import field_classifier
# from src.utils import clustering
from src.utils import llm
from src.utils import visualization

# Configure the page
st.set_page_config(
    page_title="PIA Analysis Tool",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'field_types' not in st.session_state:
    st.session_state.field_types = {}
if 'clusters' not in st.session_state:
    st.session_state.clusters = {}
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}
if 'llm_config' not in st.session_state:
    st.session_state.llm_config = {}
# if 'selected_lobs' not in st.session_state:
#     st.session_state.selected_lobs = []
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None

# Initialize filtering session state
if 'filters' not in st.session_state:
    st.session_state.filters = {}
if 'has_applied_filters' not in st.session_state:
    st.session_state.has_applied_filters = False

# Initialize LLM integration
llm_integration = llm.LLMIntegration()


# Function to reset all filters
def reset_filters():
    st.session_state.filters = {}
    if st.session_state.data is not None:
        st.session_state.filtered_data = st.session_state.data.copy()
    st.session_state.has_applied_filters = False

# Function to apply filters
def apply_filters():
    if st.session_state.data is not None:
        filtered_df = st.session_state.data.copy()
        for column, selected_values in st.session_state.filters.items():
            if selected_values and len(selected_values) > 0:
                filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
        
        st.session_state.filtered_data = filtered_df
        st.session_state.has_applied_filters = True

# Sidebar for configuration and controls
def render_sidebar():
    with st.sidebar:
        st.header("Configuration")
        
        # Data loading section
        st.subheader("Load Data")
        uploaded_file = st.file_uploader("Upload PIA Excel file", type=['xlsx', 'xls'])
        
        if uploaded_file is not None:
            df = data_loader.load_data(uploaded_file)
            if df is not None:
                original_len = len(df)
                df = df.drop_duplicates()
                df.fillna(0)
                removed = original_len - len(df)
                st.session_state.data = df
                st.success(f"Successfully loaded data with {len(df)} records and {len(df.columns)} fields")
                if removed > 0:
                    st.info(f"Removed {removed} duplicate rows")
        
        # Default data option
        if st.session_state.data is None:
            if st.button("Use default data"):
                df = data_loader.load_data("data/sample_data.xlsx", is_default=True)
                if df is not None:
                    original_len = len(df)
                    df = df.drop_duplicates()
                    df.fillna(0)
                    removed = original_len - len(df)
                    st.session_state.data = df
                    st.success(f"Successfully loaded data with {len(df)} records and {len(df.columns)} fields")
                    if removed > 0:
                        st.info(f"Removed {removed} duplicate rows")
        
        
        # LLM configuration
        st.session_state.llm_config = llm_integration.display_config_ui()
        
        # # Analysis settings
        # if st.session_state.data is not None:
        #     st.subheader("Analysis Settings")
            
        #     # Number of clusters slider
        #     n_clusters = st.slider("Number of clusters", 2, 10, 5)
            
        #     # Run analysis button
        #     if st.button("Run Analysis"):
        #         with st.spinner("Analyzing data..."):
        #             # This will trigger the analysis in the main content area
        #             st.session_state.run_analysis = True
        #             st.success("Analysis complete!")

def render_filtering_tab(df):

    if df is not None:
        # Create columns for filter controls and table
        table_col, filter_col  = st.columns([3, 1])

        with filter_col:
            for column in df.columns:
                # Get unique values for this column
                unique_values = df[column].dropna().unique().tolist()
                
                # Convert non-string values to strings for display
                display_values = [str(val) for val in unique_values]
                
                # Initialize this column's filter if not already done
                if column not in st.session_state.filters:
                    st.session_state.filters[column] = []
                
                # Create a multiselect for this column
                st.markdown(f"**{column}**")
                selected = st.multiselect(
                    f"Select values for {column}",
                    options=unique_values,
                    default=st.session_state.filters.get(column, []),
                    format_func=lambda x: str(x),
                    key=f"select_{column}"
                )
                
                # Update the filter in session state
                st.session_state.filters[column] = selected

            # Buttons for applying and clearing filters
            col1, col2 = st.columns(2)
            with col1:
                apply_button = st.button("Apply Filters", on_click=apply_filters)
            with col2:
                reset_button = st.button("Clear Filters", on_click=reset_filters)

        with table_col:
            # Display the filtered data
            if st.session_state.has_applied_filters:
                filtered_count = len(st.session_state.filtered_data)
                total_count = len(df)
                st.write(f"Showing {filtered_count} of {total_count} rows ({round(filtered_count/total_count*100, 1)}%)")
                st.dataframe(st.session_state.filtered_data, use_container_width=True)
            else:
                st.write("Showing all data (no filters applied)")
                st.dataframe(df, use_container_width=True)
    else:
        st.info("Please upload a PIA Excel file using the sidebar or use the default data if available.")



# Data Overview Tab
def render_data_overview_tab(df):
    if st.session_state.has_applied_filters:
        df = st.session_state.filtered_data

    # st.header("Descriptive Statistics")
    st.write(f"Total records: {len(df)}")
    st.write(f"Total fields: {len(df.columns)}")
    
    show_sample = st.checkbox("Show sample data")
    if show_sample:
        st.dataframe(df.head(5))

    st.markdown("---")

    # # Initialize session state for selected column
    # if 'selected_column_filter' not in st.session_state:
    #     st.session_state.selected_column_filter = None


    # st.write(f"Step 2: (Optional) Filter the Data by column")
    # # Optional filtering toggle
    # use_filter = st.checkbox(f"Filter data by `column`", value=False)

    # # If filtering is enabled
    # if use_filter:
    #     selected_column_filter = st.selectbox("Select column:", df.columns)

        
    #     # filter_column = st.selectbox("Select column to filter by", df.columns)
    #     unique_values = df[selected_column_filter].dropna().unique()
    #     selected_value = st.selectbox(f"Select a value from `{selected_column_filter}`", unique_values)

    #     # Filter the DataFrame
    #     filtered_df = df[df[selected_column_filter] == selected_value]
    #     st.write(f"Filtered data: {len(filtered_df)} rows")
    #     st.dataframe(filtered_df.head(5))
    # else:
    #     filtered_df = df

    # st.markdown("---")

    # --- Step 3: Function Selection & Execution ---
    st.write("Step 2: Choose an Analysis Function")

    selected_column_analyze = st.selectbox("Select column to analyze:", df.columns)

    # Function selector
    selected_function = st.selectbox("Select a function:", ['','Value Counts', 'Word Cloud', 'Summarize', 'Topic Model'])
    
    # Perform selected function
    if selected_function=='':
        pass
    elif selected_function == 'Value Counts':
        st.write("Value Counts")
        st.write(df[selected_column_analyze].value_counts().head(5))
    elif selected_function == 'Word Cloud':
        if pd.api.types.is_string_dtype(df[selected_column_analyze]):
            text = " ".join(df[selected_column_analyze].dropna().astype(str)).strip()

            if text:  # Check if text is not empty or whitespace
                wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)

                st.write("☁️ Word Cloud")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.warning("No valid text found in the selected column to generate a Word Cloud.")
        else:
            st.warning("Word Cloud works best with text-based columns.")
    elif selected_function == 'Summarize':
        if pd.api.types.is_string_dtype(df[selected_column_analyze]):
            text = " ".join(df[selected_column_analyze].dropna().astype(str)).strip()

            if text:  # Check if text is not empty or whitespace
                summary_key = f"column_summarizer_{selected_column_analyze}"
                if summary_key in st.session_state.summaries:
                    st.write(st.session_state.summaries[summary_key])
                else:
                    with st.spinner("Generating summary..."):
                        summary = llm_integration.generate_summary(
                            text, 
                            selected_column_analyze,
                            st.session_state.llm_config
                        )

                        st.session_state.summaries[summary_key] = summary
                        st.write(summary)
                pass
            else:
                st.warning("No valid text found in the selected column to generate a summary.")
        else:
            st.warning("LLM Summaries works best with text-based columns.")

    elif selected_function == 'Topic Model':
        if pd.api.types.is_string_dtype(df[selected_column_analyze]):
            # Check if we have text data to process
            texts = df[selected_column_analyze].dropna().astype(str).tolist()
            if texts and len(texts) > 0:
                # Allow user to configure the number of topics
                num_topics = st.slider("Number of topics:", min_value=2, max_value=10, value=3)
                
                # Topic modeling key for caching
                topic_model_key = f"topic_model_{selected_column_analyze}_{num_topics}"
                
                # Check if we already have cached results
                if topic_model_key in st.session_state:
                    topics_df, topic_assignments, embeddings, kmeans = st.session_state[topic_model_key]
                    st.write("📚 Topic Model Results")
                    st.write(topics_df)
                else:
                    with st.spinner("Generating topic model... This may take a moment."):
                        try:
                            # Call the topic modeling function
                            topics_df, topic_assignments, embeddings, kmeans = visualization.perform_topic_modeling(texts, num_topics)
                            
                            # Cache the results
                            st.session_state[topic_model_key] = (topics_df, topic_assignments, embeddings, kmeans)
                            
                            # Display the results
                            st.write("📚 Topic Model Results")
                            st.write(topics_df)
                            
                        except Exception as e:
                            st.error(str(e))
                
                # Create visualizations
                st.write("📚 Topic Visualization")
                try:
                    fig_pca = visualization.visualize_topic_clusters(embeddings, topic_assignments, num_topics, kmeans)
                    
                    # Display PCA visualization
                    st.pyplot(fig_pca)
                    
                    # # Display UMAP visualization if available
                    # if fig_umap:
                    #     st.pyplot(fig_umap)
                    # else:
                    #     st.info("Install UMAP-learn for better visualization: `pip install umap-learn`")
                        
                except Exception as e:
                    st.warning(f"Could not create visualization: {str(e)}")
                    
            else:
                st.warning("No valid text found in the selected column for topic modeling.")
        else:
            st.warning("Topic Modeling only works with text-based columns.")



    # else:
    #     st.info("No column selected yet.")


        # visualization.display_field_distribution(df, selected_column)
    
#     # Display LOB distribution if available
#     if 'Line of Business (LOB)' in df.columns:
#         lob_counts = df['Line of Business (LOB)'].value_counts().head(10)
#         fig = visualization.plot_bar_chart(
#             x=lob_counts.index, 
#             y=lob_counts.values, 
#             title='Top 10 Lines of Business',
#             x_title='Line of Business',
#             y_title='Count'
#         )
#         st.plotly_chart(fig, use_container_width=True)

# # Field Classification Tab
# def render_field_classification_tab(df):
#     st.header("Field Classification")
#     st.write("Classify fields as free text, multichoice, or other types for analysis.")
    
#     # Update field types using UI
#     st.session_state.field_types = field_classifier.display_field_classification_ui(
#         df, 
#         st.session_state.field_types
#     )

# # LOB Filtering Tab
# def render_lob_filtering_tab(df):
#     st.header("Line of Business (LOB) Filtering")
    
#     if 'Line of Business (LOB)' in df.columns:
#         lobs = sorted(df['Line of Business (LOB)'].unique().tolist())
        
#         # Option to select all LOBs
#         select_all = st.checkbox("Select All LOBs", value=len(st.session_state.selected_lobs) == len(lobs) or len(st.session_state.selected_lobs) == 0)
        
#         if select_all:
#             st.session_state.selected_lobs = lobs
#         else:
#             # Multiselect for LOBs
#             selected_lobs = st.multiselect(
#                 "Select Lines of Business", 
#                 options=lobs,
#                 default=st.session_state.selected_lobs if st.session_state.selected_lobs else None
#             )
#             st.session_state.selected_lobs = selected_lobs
        
#         # Filter data based on selected LOBs
#         if len(st.session_state.selected_lobs) > 0:
#             filtered_df = data_loader.filter_by_lob(df, st.session_state.selected_lobs)
#             st.write(f"Filtered to {len(filtered_df)} records from {len(st.session_state.selected_lobs)} LOBs")
            
#             # Preview filtered data
#             st.subheader("Filtered Data Preview")
#             st.dataframe(filtered_df.head())
            
#             # Update the filtered data in session state
#             st.session_state.filtered_data = filtered_df
#         else:
#             st.warning("No LOBs selected. All data will be used for analysis.")
#             st.session_state.filtered_data = df
#     else:
#         st.warning("No 'Line of Business (LOB)' column found in the data.")
#         st.session_state.filtered_data = df

# # Text Clustering Tab
# def render_text_clustering_tab(df):
#     st.header("Text Clustering")
    
#     # Get free text fields from classification
#     free_text_fields = field_classifier.get_fields_of_type(st.session_state.field_types, "free_text")
    
#     if free_text_fields:
#         # Select field for clustering
#         selected_field = st.selectbox("Select field for clustering", free_text_fields)
        
#         # Get filtered data based on LOB selection
#         analysis_df = st.session_state.filtered_data if st.session_state.filtered_data is not None else df
        
#         # Button to perform clustering
#         n_clusters = st.slider("Number of clusters", 2, 10, 5, key="cluster_slider")
        
#         if st.button("Perform Clustering"):
#             with st.spinner(f"Clustering '{selected_field}' into {n_clusters} clusters..."):
#                 # Get the text data
#                 text_data = analysis_df[selected_field].fillna("").tolist()
                
#                 if len(text_data) > 0:
#                     # Perform clustering
#                     clustering_results = clustering.perform_clustering(text_data, n_clusters)
                    
#                     # Store results in session state
#                     st.session_state.clusters[selected_field] = {
#                         'data': analysis_df,
#                         'cluster_labels': clustering_results['cluster_labels'],
#                         'pca_result': clustering_results['pca_result'],
#                         'preprocessed_texts': clustering_results['preprocessed_texts'],
#                         'n_clusters': n_clusters
#                     }
                    
#                     st.success(f"Successfully clustered '{selected_field}' into {n_clusters} clusters!")
#                 else:
#                     st.error("No text data available for clustering. Please check your data and filters.")
        
#         # Display clustering results if available
#         if selected_field in st.session_state.clusters:
#             st.subheader("Clustering Results")
            
#             cluster_results = st.session_state.clusters[selected_field]
#             cluster_labels = cluster_results['cluster_labels']
#             pca_result = cluster_results['pca_result']
            
#             # Visualize clusters
#             fig = visualization.plot_cluster_scatter(
#                 pca_result, 
#                 cluster_labels, 
#                 analysis_df[selected_field].fillna("").tolist(),
#                 selected_field
#             )
#             st.plotly_chart(fig, use_container_width=True)
            
#             # Display cluster statistics
#             cluster_stats = clustering.get_cluster_stats(cluster_labels, cluster_results['n_clusters'])
#             st.dataframe(cluster_stats)
            
#             # Generate cluster summaries using LLM
#             st.subheader("Cluster Summaries")
            
#             # Select a cluster to summarize
#             cluster_to_summarize = st.selectbox(
#                 "Select a cluster to summarize",
#                 [f"Cluster {c+1}" for c in range(cluster_results['n_clusters'])]
#             )
            
#             cluster_idx = int(cluster_to_summarize.split()[1]) - 1
            
#             # Get texts for the selected cluster
#             cluster_texts = clustering.get_cluster_texts(
#                 analysis_df[selected_field].fillna("").tolist(), 
#                 cluster_labels, 
#                 cluster_idx
#             )
            
#             # Button to generate summary
#             if st.button("Generate Summary"):
#                 summary_key = f"{selected_field}_{cluster_idx}"
                
#                 if summary_key in st.session_state.summaries:
#                     st.write(st.session_state.summaries[summary_key])
#                 else:
#                     with st.spinner("Generating summary..."):
#                         summary = llm_integration.generate_summary(
#                             cluster_texts, 
#                             selected_field,
#                             st.session_state.llm_config
#                         )
#                         st.session_state.summaries[summary_key] = summary
#                         st.write(summary)
            
#             # Display examples from each cluster
#             st.subheader("Cluster Examples")
            
#             selected_example_cluster = st.selectbox(
#                 "Select a cluster to view examples",
#                 [f"Cluster {c+1}" for c in range(cluster_results['n_clusters'])],
#                 key="example_cluster_selector"
#             )
            
#             example_cluster_idx = int(selected_example_cluster.split()[1]) - 1
            
#             # Get examples for the selected cluster
#             example_texts = clustering.get_cluster_texts(
#                 analysis_df[selected_field].fillna("").tolist(), 
#                 cluster_labels, 
#                 example_cluster_idx
#             )
            
#             # Display examples
#             num_examples = min(5, len(example_texts))
#             for i in range(num_examples):
#                 st.text_area(f"Example {i+1}", example_texts[i], height=100, key=f"example_{i}")
#     else:
#         st.warning("No free text fields have been classified. Please go to the Field Classification tab to identify free text fields.")

# # Analysis Results Tab
# def render_analysis_results_tab(df):
#     st.header("Analysis Results")
    
#     # Get multichoice fields
#     multichoice_fields = field_classifier.get_fields_of_type(st.session_state.field_types, "multichoice")
    
#     if multichoice_fields:
#         # Get filtered data based on LOB selection
#         analysis_df = st.session_state.filtered_data if st.session_state.filtered_data is not None else df
        
#         # Select field for analysis
#         selected_field = st.selectbox("Select multichoice field for analysis", multichoice_fields)
        
#         # Display distribution
#         value_counts = visualization.display_field_distribution(analysis_df, selected_field)
        
#         # Select a specific value to analyze
#         selected_value = st.selectbox(
#             f"Select a value from {selected_field} to analyze",
#             value_counts.index.tolist()
#         )
        
#         # Filter by selected value
#         filtered_by_value = analysis_df[analysis_df[selected_field] == selected_value]
#         st.write(f"Found {len(filtered_by_value)} records with {selected_field} = '{selected_value}'")
        
#         # Display sample of filtered data
#         st.subheader("Sample Records")
#         st.dataframe(filtered_by_value.head())
        
#         # Cross-analysis with free text fields if clusters exist
#         free_text_fields = field_classifier.get_fields_of_type(st.session_state.field_types, "free_text")
#         free_text_fields_with_clusters = [field for field in free_text_fields if field in st.session_state.clusters]
        
#         if free_text_fields_with_clusters:
#             st.subheader("Cross-Analysis with Text Clusters")
            
#             selected_text_field = st.selectbox(
#                 "Select clustered text field for cross-analysis",
#                 free_text_fields_with_clusters
#             )
            
#             if selected_text_field in st.session_state.clusters:
#                 # Get cluster labels for all data
#                 all_cluster_labels = st.session_state.clusters[selected_text_field]['cluster_labels']
#                 all_data = st.session_state.clusters[selected_text_field]['data']
                
#                 # Get cluster distribution
#                 cluster_distribution = clustering.get_cluster_distribution(
#                     filtered_by_value.index,
#                     all_data.index,
#                     all_cluster_labels,
#                     st.session_state.clusters[selected_text_field]['n_clusters']
#                 )
                
#                 if cluster_distribution is not None:
#                     # Display cluster distribution
#                     visualization.display_cluster_distribution(cluster_distribution)
#                 else:
#                     st.warning("No matching records found in the cluster data.")
        
#         # Export results
#         st.subheader("Export Results")
        
#         if st.button("Export Filtered Data to CSV"):
#             csv = filtered_by_value.to_csv(index=False)
#             st.download_button(
#                 label="Download CSV",
#                 data=csv,
#                 file_name=f"{selected_field}_{selected_value}_records.csv",
#                 mime="text/csv",
#             )
#     else:
#         st.warning("No multichoice fields have been classified. Please go to the Field Classification tab to identify multichoice fields.")

# Main app function
def main():
    # Render sidebar
    render_sidebar()
    
    # Main content area
    if st.session_state.data is None:
        st.info("Please upload a PIA Excel file using the sidebar or use the default data if available.")
    else:
        df = st.session_state.data
        
        # Display tabs for different functionalities
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Filter Data", 
            "Descriptive Statistics", 
            "Field Classification", 
            "LOB Filtering", 
            "Text Clustering", 
            "Analysis Results"
        ])
        
        # Render each tab
        # Render each tab
        with tab1:
            render_filtering_tab(df)
        with tab2:
            render_data_overview_tab(df)
        
#         with tab2:
#             render_field_classification_tab(df)
        
#         with tab3:
#             render_lob_filtering_tab(df)
        
#         with tab4:
#             render_text_clustering_tab(df)
        
#         with tab5:
#             render_analysis_results_tab(df)
    
#     # Add footer
#     st.markdown("---")
#     st.markdown("PIA Analysis Tool © 2025")

if __name__ == "__main__":
    main()
