"""
MAMMOGRAPHY RAG EXPERT SYSTEM - PRODUCTION VERSION
Fully integrated with complete_pipeline.py, real-time evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
import time
import os
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import pipeline components
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from complete_pipeline import (
    CompleteFairRAGPipeline,
    DATASET_PATH
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Mammography RAG Expert System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ADVANCED CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
/* Import modern fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* Global styles */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Main container gradient */
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-attachment: fixed;
}

.block-container {
    padding: 2rem 3rem;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.1);
}

/* Headers with gradient text */
h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3rem !important;
    margin-bottom: 1rem;
}

h2 {
    color: #2d3748;
    font-weight: 700;
    font-size: 1.8rem !important;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

h3 {
    color: #4a5568;
    font-weight: 600;
    font-size: 1.3rem !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
    width: 350px !important;  /* Add this line - default is ~300px */        
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important;
}

/* Modern buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
}

/* RAG Response Box - Premium Glass Effect */
.rag-response-box {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    backdrop-filter: blur(20px);
    border: 2px solid rgba(102, 126, 234, 0.3);
    border-radius: 20px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 
        0 10px 40px rgba(102, 126, 234, 0.2),
        inset 0 1px 0 rgba(255,255,255,0.5);
    position: relative;
    overflow: hidden;
}

.rag-response-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* Baseline Response Box - Warm Glass Effect */
.baseline-response-box {
    background: linear-gradient(135deg, rgba(251, 207, 232, 0.2) 0%, rgba(255, 159, 122, 0.2) 100%);
    backdrop-filter: blur(20px);
    border: 2px solid rgba(255, 159, 122, 0.3);
    border-radius: 20px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 
        0 10px 40px rgba(255, 159, 122, 0.2),
        inset 0 1px 0 rgba(255,255,255,0.5);
    position: relative;
    overflow: hidden;
}

.baseline-response-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #fbcfe8, #ff9f7a, #fbcfe8);
    animation: shimmer 3s infinite;
}

/* Response headers */
.response-header {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.rag-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.baseline-header {
    background: linear-gradient(135deg, #f687b3 0%, #ff9f7a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Metrics cards */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border: 1px solid rgba(102, 126, 234, 0.1);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
}

/* Status badges */
.status-badge {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
}

.status-ready {
    background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
    color: white;
}

.status-processing {
    background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
    color: white;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

/* File uploader enhancement */
[data-testid="stFileUploader"] {
    background: rgba(102, 126, 234, 0.05);
    border: 2px dashed rgba(102, 126, 234, 0.3);
    border-radius: 16px;
    padding: 2rem;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    background: rgba(102, 126, 234, 0.1);
    border-color: rgba(102, 126, 234, 0.5);
}

/* Text areas */
.stTextArea textarea {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    line-height: 1.6;
    border-radius: 12px;
    border: 2px solid rgba(102, 126, 234, 0.2);
    padding: 1rem;
}

.stTextArea textarea:focus {
    border-color: rgba(102, 126, 234, 0.5);
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

/* Metrics display */
[data-testid="metric-container"] {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border: 1px solid rgba(102, 126, 234, 0.1);
}

/* Expander styling */
.streamlit-expanderHeader {
    background: rgba(102, 126, 234, 0.05);
    border-radius: 12px;
    padding: 0.75rem;
    font-weight: 600;
}

/* Code blocks */
code {
    font-family: 'JetBrains Mono', monospace;
    background: rgba(102, 126, 234, 0.1);
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-size: 0.9rem;
}

/* Success/Error messages */
.stAlert {
    border-radius: 12px;
    border-left: 4px solid;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(102, 126, 234, 0.05);
    border-radius: 12px;
    padding: 0.5rem;
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
}

/* Loading spinner enhancement */
.stSpinner > div {
    border-color: #667eea !important;
}

/* Dividers */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.pipeline = None
    st.session_state.dataset_loaded = False
    st.session_state.current_results = None
    st.session_state.evaluation_history = []
    st.session_state.api_key = ""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_dataset():
    """Load and validate dataset"""
    try:
        df = pd.read_csv(DATASET_PATH)
        required_cols = ['Patient_ID', 'Image_ID', 'Image_Path', 'BIRADS', 
                        'Pathology', 'Findings', 'Clinical_Reports']
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.warning(f"Missing columns in dataset: {missing}")
        
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

def initialize_pipeline(api_key):
    """Initialize the complete pipeline"""
    try:
        pipeline = CompleteFairRAGPipeline(groq_api_key=api_key)
        return pipeline
    except Exception as e:
        st.error(f"Pipeline initialization failed: {e}")
        return None

def process_single_query(pipeline, image_path, query_text, pathology=None, birads=None):
    """Process a single mammography case"""
    
    # Create ground truth from inputs
    ground_truth = {
        'pathology': pathology if pathology else '',
        'birads': birads if birads else '',
        'findings': query_text
    }
    
    # Create test case
    test_case = {
        'case_id': f"manual_{datetime.now().strftime('%H%M%S')}",
        'image_path': image_path,
        'reference': query_text,  # Use query as reference for manual cases
        'ground_truth': ground_truth
    }
    
    # Process through pipeline
    result = pipeline.process_single_case(test_case)
    return result

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("# ⚙️ System Configuration")
    
    # API Key Input
    api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        value=st.session_state.api_key,
        help="Enter your Groq API key to enable LLM generation"
    )
    
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
    
    # Initialize System Button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Initialize", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter API key")
            else:
                with st.spinner("Initializing system..."):
                    pipeline = initialize_pipeline(api_key)
                    if pipeline:
                        st.session_state.pipeline = pipeline
                        st.session_state.initialized = True
                        st.success("✅ System Ready!")
                        time.sleep(1)
                        st.rerun()
    
    with col2:
        if st.button("🔄 Reset", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    st.markdown("---")
    
    # System Status
    st.markdown("## 📊 System Status")
    
    if st.session_state.initialized and st.session_state.pipeline:
        st.markdown('<span class="status-badge status-ready">SYSTEM READY</span>', 
                   unsafe_allow_html=True)
        
        # Component status
        components = {
            "Query Processor": st.session_state.pipeline.query_processor is not None,
            "Retrieval System": st.session_state.pipeline.retrieval_system is not None,
            "Generator": st.session_state.pipeline.generator is not None,
            "Evaluator": st.session_state.pipeline.evaluator is not None
        }
        
        for comp, status in components.items():
            if status:
                st.success(f"✅ {comp}")
            else:
                st.error(f"❌ {comp}")
    else:
        st.markdown('<span class="status-badge" style="background: #e53e3e;">OFFLINE</span>', 
                   unsafe_allow_html=True)
        st.info("Initialize system to begin")
    
    st.markdown("---")
    
    # Statistics
    st.markdown("## 📈 Statistics")
    st.metric("Cases Processed", len(st.session_state.evaluation_history))
    
    if st.session_state.evaluation_history:
        avg_rag = np.mean([r['rag_metrics'].get('answer_relevance', 0) 
                          for r in st.session_state.evaluation_history if r])
        avg_baseline = np.mean([r['baseline_metrics'].get('answer_relevance', 0) 
                               for r in st.session_state.evaluation_history if r])
        st.metric("Avg RAG Score", f"{avg_rag:.3f}")
        st.metric("Avg Baseline Score", f"{avg_baseline:.3f}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Header
st.markdown("# 🏥 Mammography RAG Expert System")
st.markdown("<h3 style='color: #1f77b4;'>AI-Powered Diagnostic Analysis with Evidence-Based Reasoning</h3>", unsafe_allow_html=True)

# Check system status
if not st.session_state.initialized or not st.session_state.pipeline:
    st.warning("⚠️ System not initialized. Please configure and initialize from the sidebar.")
    st.stop()

# Navigation tabs
tab1, tab2, tab3 = st.tabs(["🔬 Analyze Case", "📊 Batch Evaluation", "📈 Performance Metrics"])

# ============================================================================
# TAB 1: ANALYZE CASE
# ============================================================================

with tab1:
    st.markdown("<h2 style='color: #1f77b4;'>Single Case Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3 style='color: #1f77b4;'>🖼️ Mammography Image</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload mammography image",
            type=['png', 'jpg', 'jpeg', 'dcm'],
            help="Support for PNG, JPG, JPEG, and DICOM formats"
        )
        
        if uploaded_file:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Mammography", use_container_width=200)
    
    with col2:
        st.markdown("<h3 style='color: #1f77b4;'>📝 Clinical Query</h3>", unsafe_allow_html=True)
        query = st.text_area(
            "Enter diagnostic query",
            height=150,
            placeholder="Example: Analyze this mammogram for pathology assessment and provide BI-RADS classification",
            help="Describe what you want to analyze in the mammography"
        )
        
        # Optional clinical context
        with st.expander("🔍 Add Clinical Context (Optional)"):
            col_a, col_b = st.columns(2)
            with col_a:
                pathology = st.selectbox(
                    "Known Pathology",
                    ["Unknown", "Benign", "Malignant", "Normal"]
                )
            with col_b:
                birads = st.selectbox(
                    "BI-RADS Category",
                    ["Unknown", "1", "2", "3", "4", "5", "6"]
                )
            
            additional_findings = st.text_input(
                "Additional Findings",
                placeholder="e.g., dense breast tissue, calcifications"
            )
    
    # Analyze button
    if st.button("🔬 ANALYZE", type="primary", use_container_width=True):
        if not uploaded_file:
            st.error("Please upload an image")
        elif not query:
            st.error("Please enter a query")
        else:
            # Save temporary image
            temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Process with loading animation
                with st.spinner("🔄 Processing mammography analysis..."):
                    
                    # Progress indicators
                    progress = st.progress(0)
                    status = st.empty()
                    
                    status.text("📊 Processing query...")
                    progress.progress(20)
                    time.sleep(0.5)
                    
                    status.text("🔍 Retrieving similar cases...")
                    progress.progress(40)
                    
                    # Process through pipeline
                    result = process_single_query(
                        st.session_state.pipeline,
                        temp_path,
                        query,
                        pathology if pathology != "Unknown" else None,
                        birads if birads != "Unknown" else None
                    )
                    
                    status.text("🤖 Generating responses...")
                    progress.progress(60)
                    
                    status.text("📈 Evaluating performance...")
                    progress.progress(80)
                    
                    status.text("✅ Analysis complete!")
                    progress.progress(100)
                    time.sleep(0.5)
                    
                    # Store results
                    if result:
                        st.session_state.current_results = result
                        st.session_state.evaluation_history.append(result)
                    
                    # Clean up
                    progress.empty()
                    status.empty()
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    # Display results if available
    if st.session_state.current_results:
        st.markdown("---")
        st.markdown("## 📋 Analysis Results")
        
        result = st.session_state.current_results
        
        # Response comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="rag-response-box">', unsafe_allow_html=True)
            st.markdown('<div class="response-header rag-header">🤖 RAG-Enhanced Response</div>', 
                       unsafe_allow_html=True)
            st.markdown(result['rag_response'])
            
            # Show retrieval stats
            if 'retrieval_stats' in result:
                stats = result['retrieval_stats']
                st.caption(f"📚 Retrieved {stats['n_contexts']} similar cases")
                st.caption(f"🎯 Alignment: {stats['pathology_alignment']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="baseline-response-box">', unsafe_allow_html=True)
            st.markdown('<div class="response-header baseline-header">📖 Baseline Response</div>', 
                       unsafe_allow_html=True)
            st.markdown(result['baseline_response'])
            st.caption("💡 Generated without retrieval")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Metrics visualization
        st.markdown("### 📊 Performance Metrics")
        
        # Prepare metrics data
        metrics_df = pd.DataFrame([
            {
                'Metric': 'Answer Relevance',
                'RAG': result['rag_metrics']['answer_relevance'],
                'Baseline': result['baseline_metrics']['answer_relevance']
            },
            
            {
                'Metric': 'Clinical Coherence',
                'RAG': result['rag_metrics']['clinical_coherence'],
                'Baseline': result['baseline_metrics']['clinical_coherence']
            },
            {
                'Metric': 'Semantic Similarity',
                'RAG': result['rag_metrics']['semantic_similarity'],
                'Baseline': result['baseline_metrics']['semantic_similarity']
            }
        ])
        
        # Create interactive bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='RAG',
            x=metrics_df['Metric'],
            y=metrics_df['RAG'],
            marker=dict(
                color='rgba(102, 126, 234, 0.8)',
                line=dict(color='rgba(102, 126, 234, 1)', width=2)
            ),
            text=metrics_df['RAG'].round(3),
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Baseline',
            x=metrics_df['Metric'],
            y=metrics_df['Baseline'],
            marker=dict(
                color='rgba(255, 159, 122, 0.8)',
                line=dict(color='rgba(255, 159, 122, 1)', width=2)
            ),
            text=metrics_df['Baseline'].round(3),
            textposition='outside'
        ))
        
        fig.update_layout(
            barmode='group',
            yaxis=dict(range=[0, 1.1], title='Score'),
            xaxis=dict(title=''),
            template='plotly_white',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        rag_avg = np.mean([
            result['rag_metrics']['answer_relevance'],
            
            result['rag_metrics']['clinical_coherence'],
            result['rag_metrics']['semantic_similarity']
        ])
        
        baseline_avg = np.mean([
            result['baseline_metrics']['answer_relevance'],
            
            result['baseline_metrics']['clinical_coherence'],
            result['baseline_metrics']['semantic_similarity']
        ])
        
        with col1:
            st.metric("RAG Average", f"{rag_avg:.3f}")
        with col2:
            st.metric("Baseline Average", f"{baseline_avg:.3f}")
        with col3:
            improvement = ((rag_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
            st.metric("Improvement", f"{improvement:+.1f}%")
        with col4:
            winner = "🏆 RAG" if rag_avg > baseline_avg else "🏆 Baseline"
            st.metric("Winner", winner)

# ============================================================================
# TAB 2: BATCH EVALUATION
# ============================================================================

with tab2:
    st.markdown("## Batch Evaluation")
    st.markdown("Run comprehensive evaluation on multiple test cases from the dataset")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_cases = st.number_input(
            "Number of test cases",
            min_value=5,
            max_value=100,
            value=30,
            step=5
        )
    
    with col2:
        test_type = st.selectbox(
            "Test case selection",
            ["Balanced (Equal pathology types)", "Random", "Challenging cases"]
        )
    
    with col3:
        st.markdown("")
        st.markdown("")
        run_batch = st.button("🚀 Run Batch Evaluation", type="primary", use_container_width=True)
    
    if run_batch:
        with st.spinner(f"Running evaluation on {n_cases} cases..."):
            try:
                # Create progress tracking
                progress = st.progress(0)
                status = st.empty()
                
                # Run evaluation
                results = st.session_state.pipeline.run_comprehensive_evaluation(n_cases)
                
                progress.progress(100)
                status.success("✅ Evaluation complete!")
                time.sleep(1)
                progress.empty()
                status.empty()
                
                # Display results
                if results and 'overall_comparison' in results:
                    st.markdown("### 📊 Evaluation Results")
                    
                    overall = results['overall_comparison']
                    
                    # Overall metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RAG Performance", f"{overall['rag_mean']:.3f}")
                    with col2:
                        st.metric("Baseline Performance", f"{overall['baseline_mean']:.3f}")
                    with col3:
                        st.metric("Improvement", f"{overall['percent_improvement']:+.1f}%")
                    
                    # Statistical significance
                    if 'p_value' in overall:
                        if overall['significant']:
                            st.success(f"✅ Statistically significant (p={overall['p_value']:.4f})")
                        else:
                            st.info(f"📊 Not statistically significant (p={overall['p_value']:.4f})")
                        
                        st.info(f"Effect size: {overall['effect_size']} (Cohen's d = {overall['cohens_d']:.3f})")
                    
                    # Detailed metrics
                    if 'core_metrics' in results:
                        st.markdown("### Detailed Metrics Analysis")
                        
                        metrics_data = []
                        for metric, values in results['core_metrics'].items():
                            metrics_data.append({
                                'Metric': metric.replace('_', ' ').title(),
                                'RAG Mean': values['rag_mean'],
                                'RAG Std': values['rag_std'],
                                'Baseline Mean': values['baseline_mean'],
                                'Baseline Std': values['baseline_std'],
                                'Difference': values['difference'],
                                'P-Value': values.get('p_value', 'N/A')
                            })
                        
                        df = pd.DataFrame(metrics_data)
                        st.dataframe(df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")

# ============================================================================
# TAB 3: PERFORMANCE METRICS
# ============================================================================

with tab3:
    st.markdown("## Performance Analytics")
    
    if not st.session_state.evaluation_history:
        st.info("No evaluation data available. Process some cases to see analytics.")
    else:
        # Calculate aggregate metrics
        history = st.session_state.evaluation_history
        
        # Time series data
        st.markdown("### 📈 Performance Trends")
        
        # Prepare time series data
        timestamps = list(range(1, len(history) + 1))
        rag_scores = []
        baseline_scores = []
        
        for result in history:
            if result and 'rag_metrics' in result and 'baseline_metrics' in result:
                # Calculate average scores
                rag_avg = np.mean([
                    result['rag_metrics'].get('answer_relevance', 0),
                    
                    result['rag_metrics'].get('clinical_coherence', 0),
                    result['rag_metrics'].get('semantic_similarity', 0)
                ])
                baseline_avg = np.mean([
                    result['baseline_metrics'].get('answer_relevance', 0),
                    
                    result['baseline_metrics'].get('clinical_coherence', 0),
                    result['baseline_metrics'].get('semantic_similarity', 0)
                ])
                
                rag_scores.append(rag_avg)
                baseline_scores.append(baseline_avg)
        
        if rag_scores and baseline_scores:
            # Create line chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=rag_scores,
                mode='lines+markers',
                name='RAG System',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8, color='#667eea'),
                fill='tonexty',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=baseline_scores,
                mode='lines+markers',
                name='Baseline System',
                line=dict(color='#ff9f7a', width=3),
                marker=dict(size=8, color='#ff9f7a'),
                fill='tonexty',
                fillcolor='rgba(255, 159, 122, 0.1)'
            ))
            
            fig.update_layout(
                title="Performance Over Time",
                xaxis_title="Case Number",
                yaxis_title="Average Score",
                yaxis=dict(range=[0, 1]),
                template='plotly_white',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.markdown("### 📊 Summary Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### RAG System")
                rag_stats = pd.DataFrame({
                    'Metric': ['Mean', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{np.mean(rag_scores):.3f}",
                        f"{np.std(rag_scores):.3f}",
                        f"{np.min(rag_scores):.3f}",
                        f"{np.max(rag_scores):.3f}"
                    ]
                })
                st.dataframe(rag_stats, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### Baseline System")
                baseline_stats = pd.DataFrame({
                    'Metric': ['Mean', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{np.mean(baseline_scores):.3f}",
                        f"{np.std(baseline_scores):.3f}",
                        f"{np.min(baseline_scores):.3f}",
                        f"{np.max(baseline_scores):.3f}"
                    ]
                })
                st.dataframe(baseline_stats, use_container_width=True, hide_index=True)
            
            # Metric breakdown
            st.markdown("### 🔍 Metric Breakdown")
            
            # Collect all metrics
            metric_names = ['answer_relevance',  
                          'clinical_coherence', 'semantic_similarity']
            
            rag_metric_data = {metric: [] for metric in metric_names}
            baseline_metric_data = {metric: [] for metric in metric_names}
            
            for result in history:
                if result and 'rag_metrics' in result and 'baseline_metrics' in result:
                    for metric in metric_names:
                        rag_metric_data[metric].append(result['rag_metrics'].get(metric, 0))
                        baseline_metric_data[metric].append(result['baseline_metrics'].get(metric, 0))
            
            # Create radar chart
            fig = go.Figure()
            
            categories = [m.replace('_', ' ').title() for m in metric_names]
            
            rag_values = [np.mean(rag_metric_data[m]) for m in metric_names]
            baseline_values = [np.mean(baseline_metric_data[m]) for m in metric_names]
            
            fig.add_trace(go.Scatterpolar(
                r=rag_values,
                theta=categories,
                fill='toself',
                name='RAG System',
                line=dict(color='#667eea', width=2),
                fillcolor='rgba(102, 126, 234, 0.3)'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=baseline_values,
                theta=categories,
                fill='toself',
                name='Baseline System',
                line=dict(color='#ff9f7a', width=2),
                fillcolor='rgba(255, 159, 122, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Metric Performance Comparison",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance distribution
            st.markdown("### 📊 Score Distribution")
            
            # Create box plots
            fig = go.Figure()
            
            for metric in metric_names:
                fig.add_trace(go.Box(
                    y=rag_metric_data[metric],
                    name=metric.replace('_', ' ').title(),
                    marker=dict(color='#667eea'),
                    boxmean=True
                ))
            
            fig.update_layout(
                title="RAG System - Metric Distribution",
                yaxis_title="Score",
                showlegend=False,
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export data option
            st.markdown("### 💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Prepare export data
                export_data = []
                for i, result in enumerate(history, 1):
                    if result:
                        export_data.append({
                            'Case': i,
                            'RAG_Relevance': result['rag_metrics'].get('answer_relevance', 0),
                           
                            'RAG_Coherence': result['rag_metrics'].get('clinical_coherence', 0),
                            'RAG_Similarity': result['rag_metrics'].get('semantic_similarity', 0),
                            'Baseline_Relevance': result['baseline_metrics'].get('answer_relevance', 0),
                            
                            'Baseline_Coherence': result['baseline_metrics'].get('clinical_coherence', 0),
                            'Baseline_Similarity': result['baseline_metrics'].get('semantic_similarity', 0)
                        })
                
                if export_data:
                    df_export = pd.DataFrame(export_data)
                    csv = df_export.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                    st.session_state.evaluation_history = []
                    st.session_state.current_results = None
                    st.success("History cleared!")
                    time.sleep(1)
                    st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #718096; padding: 2rem;'>
        <p><strong>Mammography RAG Expert System v3.0</strong></p>
        <p>Powered by Llama 4 Scout 17B • Advanced Retrieval-Augmented Generation</p>
        <p style='font-size: 0.9rem; margin-top: 1rem;'>
            ⚠️ For Research and Educational Purposes Only
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

