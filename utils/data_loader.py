"""
Data loading utilities for the PIA Analysis Tool.
"""

import pandas as pd
import streamlit as st
from pathlib import Path

def load_data(source, is_default=False):
    """
    Unified function to load data from various sources.
    
    Args:
        source: Path string, Path object, or Streamlit uploaded file object
        is_default: Boolean indicating if this is loading the default data
        
    Returns:
        DataFrame: Loaded data or None if loading failed
    """
    try:
        # Handle Streamlit uploaded file
        if hasattr(source, 'read'):
            df = pd.read_excel(source)
            return df
        
        # Handle Path object or string path
        path = Path(source) if not isinstance(source, Path) else source
        
        if path.exists():
            df = pd.read_excel(path)
            return df
        else:
            message = f"Default data file not found at {path}" if is_default else f"File not found at {path}"
            st.warning(message)
            return None
            
    except Exception as e:
        source_type = "default data" if is_default else "file"
        st.error(f"Error loading {source_type}: {e}")
        return None

def filter_by_lob(df, selected_lobs):
    """
    Filter DataFrame by selected Lines of Business.
    
    Args:
        df: DataFrame to filter
        selected_lobs: List of selected LOBs
        
    Returns:
        DataFrame: Filtered data
    """
    if 'Line of Business (LOB)' in df.columns and selected_lobs:
        return df[df['Line of Business (LOB)'].isin(selected_lobs)]
    return df
