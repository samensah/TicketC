"""
Visualization utilities for the PIA Analysis Tool.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def plot_pie_chart(names, values, title):
    """
    Create a pie chart.
    
    Args:
        names: Names for the pie chart slices
        values: Values for the pie chart slices
        title: Chart title
        
    Returns:
        fig: Plotly figure
    """
    fig = px.pie(
        names=names,
        values=values,
        title=title
    )
    return fig

def plot_bar_chart(x, y, title, x_title="", y_title="", color=None):
    """
    Create a bar chart.
    
    Args:
        x: X-axis values
        y: Y-axis values
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        color: Column to use for coloring
        
    Returns:
        fig: Plotly figure
    """
    if color is not None:
        fig = px.bar(
            x=x,
            y=y,
            title=title,
            color=color,
            color_discrete_sequence=px.colors.qualitative.G10
        )
    else:
        fig = px.bar(
            x=x,
            y=y,
            title=title
        )
    
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title
    )
    
    return fig

def plot_cluster_scatter(pca_result, cluster_labels, texts, field_name):
    """
    Create a scatter plot of clusters.
    
    Args:
        pca_result: PCA result for 2D visualization
        cluster_labels: Cluster labels
        texts: Original texts
        field_name: Name of the field being clustered
        
    Returns:
        fig: Plotly figure
    """
    # Create a DataFrame for visualization
    viz_df = pd.DataFrame({
        'x': pca_result[:, 0],
        'y': pca_result[:, 1],
        'cluster': [f"Cluster {c+1}" for c in cluster_labels],
        'text': texts
    })
    
    # Plot clusters
    fig = px.scatter(
        viz_df, 
        x='x', 
        y='y', 
        color='cluster',
        hover_data=['text'],
        title=f"Clusters for '{field_name}'",
        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'},
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    return fig

def display_field_distribution(df, field_name):
    """
    Display the distribution of values in a field.
    
    Args:
        df: DataFrame containing the data
        field_name: Name of the field to analyze
    """
    # Get value counts
    value_counts = df[field_name].value_counts()
    
    # Display distribution
    st.subheader(f"Distribution of {field_name}")
    fig = plot_pie_chart(
        names=value_counts.index,
        values=value_counts.values,
        title=f"Distribution of {field_name}"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    return value_counts

def display_cluster_distribution(cluster_distribution):
    """
    Display the distribution of clusters.
    
    Args:
        cluster_distribution: DataFrame containing cluster distribution
    """
    fig = plot_bar_chart(
        x=cluster_distribution['Cluster'],
        y=cluster_distribution['Count'],
        title=f"Cluster Distribution",
        x_title="Cluster",
        y_title="Count",
        color="Cluster"
    )
    st.plotly_chart(fig, use_container_width=True)


def perform_topic_modeling(texts, num_topics=3):
    """
    Perform topic modeling using sentence transformers and KMeans clustering.

    Args:
        texts (list): List of text strings
        num_topics (int): Number of topics to extract

    Returns:
        tuple: (topics_df, topic_assignments, embeddings, kmeans_model)
    """
    try:
        # Imports
        import numpy as np
        import pandas as pd
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import CountVectorizer

        # Load transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Clean and prepare input
        texts = [str(t).strip() for t in texts if isinstance(t, str) and t.strip()]
        if len(texts) < num_topics or len(texts) < 5:
            raise ValueError("Not enough documents for meaningful topic modeling.")

        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False)

        # KMeans clustering
        kmeans = KMeans(n_clusters=num_topics, random_state=42, n_init=10)
        topic_assignments = kmeans.fit_predict(embeddings)

        # Initialize topic metadata
        topics_summary = []

        for topic_idx in range(num_topics):
            # Collect texts assigned to this topic
            topic_texts = [texts[i] for i, label in enumerate(topic_assignments) if label == topic_idx]

            # Skip empty or very small topics
            if len(topic_texts) < 2:
                continue

            # Combine all text for the topic
            topic_text = " ".join(topic_texts)

            # Vectorize with conservative thresholds for small samples
            vectorizer = CountVectorizer(
                stop_words='english',
                max_features=20,
                max_df=1.0,
                min_df=1
            )

            word_counts = vectorizer.fit_transform([topic_text])
            feature_names = vectorizer.get_feature_names_out()

            if len(feature_names) == 0:
                top_words = ["(no keywords)"]
            else:
                word_freq = word_counts.toarray()[0]
                top_words_idx = word_freq.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]

            # Sample a short snippet from topic
            sample_text = topic_texts[0]
            sample_snippet = sample_text[:150] + "..." if len(sample_text) > 150 else sample_text

            topic_weight = round(len(topic_texts) / len(texts) * 100, 2)

            topics_summary.append({
                "Topic": f"Topic {topic_idx + 1}",
                # "Weight (%)": topic_weight,
                "Top Words": ", ".join(top_words),
                "Sample Document": sample_snippet
            })

        # Build DataFrame
        topics_df = pd.DataFrame(topics_summary)

        return topics_df, topic_assignments, embeddings, kmeans

    except Exception as e:
        raise Exception(f"Error in topic modeling: {str(e)}")


def visualize_topic_clusters(embeddings, topic_assignments, num_topics, kmeans):
    """
    Create visualizations for topic modeling results.
    
    Args:
        embeddings (np.array): Document embeddings
        topic_assignments (np.array): Topic assignments for each document
        num_topics (int): Number of topics
        kmeans (KMeans): Fitted KMeans model
    
    Returns:
        tuple: (pca_figure, umap_figure) - Matplotlib figures for visualizations
    """
    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Reduce dimensions to 2D for visualization using PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Create scatter plot
        fig_pca, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            c=topic_assignments, 
            cmap='Set1', 
            alpha=0.6, 
            s=50
        )
        
        # Add cluster centers
        centers_2d = pca.transform(kmeans.cluster_centers_)
        ax.scatter(
            centers_2d[:, 0], 
            centers_2d[:, 1], 
            c='black', 
            marker='x', 
            s=200, 
            linewidths=3, 
            label='Cluster Centers'
        )
        
        # Add legend
        legend_labels = [f"Topic {i+1}" for i in range(num_topics)]
        legend_handles = [plt.scatter([], [], c=plt.cm.Set1(i/10), s=50, alpha=0.6) for i in range(num_topics)]
        ax.legend(legend_handles, legend_labels, loc='best')
        
        plt.title("Document Clusters Visualization (PCA)")
        # plt.xlabel("First Principal Component")
        # plt.ylabel("Second Principal Component")
        plt.tight_layout()
        
        # # Try UMAP visualization if available
        # fig_umap = None
        # try:
        #     # import umap
        #     import umap.umap_ as umap

        #     reducer = umap.UMAP(n_components=2, random_state=42)
        #     embeddings_umap = reducer.fit_transform(embeddings)
            
        #     fig_umap, ax2 = plt.subplots(figsize=(10, 8))
        #     scatter2 = ax2.scatter(
        #         embeddings_umap[:, 0], 
        #         embeddings_umap[:, 1], 
        #         c=topic_assignments, 
        #         cmap='Set1', 
        #         alpha=0.6, 
        #         s=50
        #     )
            
        #     ax2.legend(legend_handles, legend_labels, loc='best')
        #     plt.title("Document Clusters Visualization (UMAP)")
        #     plt.xlabel("UMAP Component 1")
        #     plt.ylabel("UMAP Component 2")
        #     plt.tight_layout()
            
        # except ImportError:
        #     pass
        
        return fig_pca
        
    except Exception as e:
        raise Exception(f"Error in visualization: {str(e)}")
    
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_excel("../../data/sample_data.xlsx")
    texts = list(df.Description)
    topics_df, topic_assignments, embeddings, kmeans = perform_topic_modeling(texts=texts, num_topics=3)
    print(kmeans)

