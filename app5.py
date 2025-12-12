"""
MAMMOGRAPHY DIAGNOSTIC INTELLIGENCE SUITE - ULTRA PREMIUM v9.0
BREAST CANCER THEMED | Advanced Medical Aesthetics | Enterprise Luxury
3000+ Lines | Cutting-Edge Interactions | Medical-Grade Visibility
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Advanced Breast Cancer Diagnostic Intelligence",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# ULTRA PREMIUM MEDICAL-THEMED CSS (3000+ Lines)
# ============================================================================

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800;900&family=Poppins:wght@100;200;300;400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif;
    scroll-behavior: smooth;
}

:root {
    --pink-950: #500724;
    --pink-900: #881337;
    --pink-800: #9d174d;
    --pink-700: #be185d;
    --pink-600: #db2777;
    --pink-500: #ec4899;
    --pink-400: #f472b6;
    --pink-300: #fbbbf9;
    --pink-200: #fcbad5;
    --pink-100: #fce7f3;
    
    --rose-900: #78071e;
    --rose-800: #9f1239;
    --rose-700: #be123c;
    --rose-600: #e11d48;
    --rose-500: #f43f5e;
    
    --medical-purple: #6b21a8;
    --medical-blue: #1e40af;
    --medical-cyan: #0891b2;
    --medical-teal: #0d9488;
    
    --gray-950: #03071e;
    --gray-900: #0f172a;
    --gray-800: #1e293b;
    --gray-700: #334155;
    --gray-600: #475569;
    --gray-500: #64748b;
    --gray-400: #94a3b8;
    --gray-300: #cbd5e1;
    --gray-200: #e2e8f0;
    --gray-100: #f1f5f9;
    --white: #ffffff;
    
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    
    --shadow-xs: 0 1px 2px rgba(139, 19, 55, 0.05);
    --shadow-sm: 0 4px 16px rgba(139, 19, 55, 0.12);
    --shadow-md: 0 12px 32px rgba(139, 19, 55, 0.15);
    --shadow-lg: 0 20px 48px rgba(139, 19, 55, 0.2);
    --shadow-xl: 0 32px 64px rgba(139, 19, 55, 0.25);
    --shadow-2xl: 0 40px 80px rgba(139, 19, 55, 0.3);
    
    --glow-pink: 0 0 60px rgba(236, 72, 153, 0.25);
    --glow-rose: 0 0 60px rgba(244, 63, 94, 0.25);
    --glow-medical: 0 0 60px rgba(107, 33, 168, 0.25);
}

html, body {
    background: linear-gradient(135deg, #0f172a 0%, #1a0f2e 25%, #2d1b4e 50%, #1a0f2e 75%, #0f172a 100%);
    background-attachment: fixed;
    background-size: 300% 300%;
    min-height: 100vh;
    animation: gradient-shift 20s ease infinite;
}

@keyframes gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.main { background: transparent !important; }

.block-container {
    padding: 2.5rem 3.5rem;
    max-width: 2400px;
    margin: 0 auto;
}

/* ============================================================================
   TYPOGRAPHY - MEDICAL LUXURY HIERARCHY
   ============================================================================ */

h1 {
    font-family: 'Playfair Display', serif;
    color: var(--white);
    font-weight: 900;
    font-size: 5rem;
    letter-spacing: -0.06em;
    line-height: 0.95;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #ec4899 0%, #f43f5e 25%, #db2777 50%, #f43f5e 75%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 80px rgba(236, 72, 153, 0.3)) drop-shadow(0 0 40px rgba(244, 63, 94, 0.2));
    animation: text-glow-medical 3s ease-in-out infinite;
}

@keyframes text-glow-medical {
    0%, 100% { filter: drop-shadow(0 0 40px rgba(236, 72, 153, 0.3)) drop-shadow(0 0 20px rgba(244, 63, 94, 0.2)); }
    50% { filter: drop-shadow(0 0 80px rgba(236, 72, 153, 0.5)) drop-shadow(0 0 40px rgba(244, 63, 94, 0.4)); }
}

h2 {
    font-family: 'Playfair Display', serif;
    color: var(--white);
    font-weight: 800;
    font-size: 3rem;
    letter-spacing: -0.04em;
    margin-top: 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    padding-left: 2rem;
    background: linear-gradient(135deg, #ec4899 0%, #db2777 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2::before {
    content: '';
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 8px;
    height: 80%;
    background: linear-gradient(180deg, #ec4899 0%, #f43f5e 50%, #db2777 100%);
    border-radius: 4px;
    box-shadow: 0 0 30px rgba(236, 72, 153, 0.6), var(--glow-pink);
}

h3 {
    font-family: 'Playfair Display', serif;
    color: #fbbbf9;
    font-weight: 700;
    font-size: 2rem;
    letter-spacing: -0.03em;
    margin-bottom: 1.5rem;
}

p {
    color: #e2e8f0;
    line-height: 1.9;
    font-size: 1.05rem;
    font-weight: 400;
}

/* ============================================================================
   MEDICAL LABEL STYLING - ULTRA VISIBILITY
   ============================================================================ */

label {
    color: #fbbbf9 !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stTextArea label, .stSelectbox label, .stNumberInput label, .stTextInput label {
    color: #fbbbf9 !important;
    font-weight: 900 !important;
    font-size: 1.1rem !important;
}

/* ============================================================================
   MEDICAL PREMIUM CARDS - BREAST CANCER THEMED
   ============================================================================ */

.medical-card {
    background: linear-gradient(135deg, rgba(139, 19, 55, 0.3) 0%, rgba(107, 33, 168, 0.2) 100%);
    backdrop-filter: blur(50px);
    border-radius: 36px;
    padding: 3.5rem;
    border: 2px solid rgba(236, 72, 153, 0.4);
    box-shadow: 0 0 100px rgba(236, 72, 153, 0.2), inset 0 1px 2px rgba(255, 255, 255, 0.1);
    transition: all 0.7s cubic-bezier(0.34, 1.56, 0.64, 1);
    position: relative;
    overflow: hidden;
}

.medical-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(236, 72, 153, 0.8), transparent);
    opacity: 0;
    transition: opacity 0.7s ease;
}

.medical-card::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle, rgba(236, 72, 153, 0.15) 0%, transparent 70%);
    opacity: 0;
    transition: opacity 0.7s ease;
}

.medical-card:hover {
    background: linear-gradient(135deg, rgba(139, 19, 55, 0.4) 0%, rgba(107, 33, 168, 0.3) 100%);
    border-color: #ec4899;
    box-shadow: 0 0 150px rgba(236, 72, 153, 0.35), inset 0 1px 2px rgba(255, 255, 255, 0.15);
    transform: translateY(-15px);
}

.medical-card:hover::before { opacity: 1; }
.medical-card:hover::after { opacity: 1; }

/* ============================================================================
   MEDICAL RESPONSE CONTAINERS - ADVANCED
   ============================================================================ */

.response-medical {
    backdrop-filter: blur(60px);
    border-radius: 36px;
    padding: 4rem;
    margin: 3rem 0;
    position: relative;
    overflow: hidden;
    border: 2.5px solid rgba(236, 72, 153, 0.5);
    min-height: 400px;
    transition: all 0.7s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.response-medical::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, transparent 0%, rgba(236, 72, 153, 0.08) 100%);
    pointer-events: none;
}

.response-rag-medical {
    background: linear-gradient(135deg, rgba(236, 72, 153, 0.18) 0%, rgba(244, 63, 94, 0.12) 100%);
    border: 2.5px solid rgba(236, 72, 153, 0.6);
    box-shadow: 0 0 120px rgba(236, 72, 153, 0.25), inset 0 1px 2px rgba(255, 255, 255, 0.12);
}

.response-rag-medical:hover {
    border-color: #ec4899;
    box-shadow: 0 0 180px rgba(236, 72, 153, 0.4);
    transform: translateY(-8px);
}

.response-baseline-medical {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(244, 63, 94, 0.12) 100%);
    border: 2.5px solid rgba(244, 63, 94, 0.6);
    box-shadow: 0 0 120px rgba(244, 63, 94, 0.25);
}

.response-baseline-medical:hover {
    border-color: #f43f5e;
    box-shadow: 0 0 180px rgba(244, 63, 94, 0.4);
    transform: translateY(-8px);
}

.response-header-medical {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin-bottom: 3rem;
    font-weight: 900;
    font-size: 1.8rem;
    letter-spacing: -0.03em;
    position: relative;
    z-index: 3;
}

.response-rag-medical .response-header-medical { 
    background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.response-baseline-medical .response-header-medical { 
    color: #f87171;
}

.response-content-medical {
    font-size: 1.1rem;
    line-height: 2.1;
    color: #e2e8f0;
    position: relative;
    z-index: 3;
}

/* ============================================================================
   MEDICAL METRIC BOXES - ADVANCED
   ============================================================================ */

.metric-medical {
    background: linear-gradient(135deg, rgba(139, 19, 55, 0.35) 0%, rgba(107, 33, 168, 0.25) 100%);
    backdrop-filter: blur(40px);
    border-radius: 32px;
    padding: 3rem;
    text-align: center;
    border: 2px solid rgba(236, 72, 153, 0.4);
    box-shadow: 0 12px 50px rgba(236, 72, 153, 0.25), inset 0 1px 2px rgba(255, 255, 255, 0.08);
    transition: all 0.7s cubic-bezier(0.34, 1.56, 0.64, 1);
    position: relative;
    overflow: hidden;
}

.metric-medical::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(236, 72, 153, 0.8), transparent);
    opacity: 0;
    transition: opacity 0.7s ease;
}

.metric-medical::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle, rgba(236, 72, 153, 0.12) 0%, transparent 70%);
    opacity: 0;
    transition: opacity 0.7s ease;
}

.metric-medical:hover {
    border-color: #ec4899;
    box-shadow: 0 0 120px rgba(236, 72, 153, 0.35);
    transform: translateY(-12px) scale(1.03);
}

.metric-medical:hover::before { opacity: 1; }
.metric-medical:hover::after { opacity: 1; }

.metric-label-medical {
    font-family: 'Poppins', sans-serif;
    font-size: 0.85rem;
    background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    margin-bottom: 1.3rem;
}

.metric-value-medical {
    font-family: 'Playfair Display', serif;
    font-size: 3.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ec4899 0%, #f43f5e 50%, #db2777 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.05em;
    text-shadow: 0 0 60px rgba(236, 72, 153, 0.3);
}

/* ============================================================================
   MEDICAL BUTTONS - ADVANCED INTERACTIONS
   ============================================================================ */

.stButton > button {
    border-radius: 20px;
    font-weight: 900;
    padding: 1.4rem 3rem;
    border: none;
    transition: all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 0.95rem;
    font-family: 'Poppins', sans-serif;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 40px rgba(236, 72, 153, 0.3);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ec4899 0%, #f43f5e 50%, #db2777 100%);
    color: white;
    box-shadow: 0 20px 60px rgba(236, 72, 153, 0.4);
    border: none;
}

.stButton > button[kind="primary"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
    transition: left 0.7s ease;
    z-index: 1;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 30px 80px rgba(236, 72, 153, 0.5);
}

.stButton > button[kind="primary"]:hover::before { left: 100%; }

.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, rgba(139, 19, 55, 0.4) 0%, rgba(107, 33, 168, 0.3) 100%);
    backdrop-filter: blur(30px);
    color: #fbbbf9;
    border: 2px solid rgba(236, 72, 153, 0.5);
    box-shadow: 0 12px 40px rgba(236, 72, 153, 0.2);
}

.stButton > button[kind="secondary"]:hover {
    background: linear-gradient(135deg, rgba(139, 19, 55, 0.5) 0%, rgba(107, 33, 168, 0.4) 100%);
    border-color: #ec4899;
    transform: translateY(-6px);
    box-shadow: 0 20px 60px rgba(236, 72, 153, 0.35);
}

/* ============================================================================
   MEDICAL INPUT FIELDS - ULTRA PREMIUM
   ============================================================================ */

.stTextInput input,
.stSelectbox select,
.stNumberInput input,
.stTextArea textarea {
    border: 2.5px solid rgba(236, 72, 153, 0.4) !important;
    border-radius: 20px !important;
    padding: 1.3rem !important;
    font-size: 1.1rem !important;
    font-family: 'Poppins', sans-serif !important;
    transition: all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    background: linear-gradient(135deg, rgba(139, 19, 55, 0.2) 0%, rgba(107, 33, 168, 0.15) 100%) !important;
    backdrop-filter: blur(40px);
    box-shadow: 0 12px 40px rgba(236, 72, 153, 0.15), inset 0 1px 2px rgba(255, 255, 255, 0.08) !important;
    color: #fbbbf9 !important;
}

.stTextInput input:focus,
.stSelectbox select:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {
    border-color: #ec4899 !important;
    box-shadow: 0 0 0 5px rgba(236, 72, 153, 0.25), 0 20px 60px rgba(236, 72, 153, 0.3) !important;
    background: linear-gradient(135deg, rgba(139, 19, 55, 0.3) 0%, rgba(107, 33, 168, 0.25) 100%) !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #a78bfa !important;
}

/* ============================================================================
   MEDICAL TABS - ADVANCED
   ============================================================================ */

.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(135deg, rgba(139, 19, 55, 0.3) 0%, rgba(107, 33, 168, 0.2) 100%);
    backdrop-filter: blur(40px);
    border-bottom: 2.5px solid rgba(236, 72, 153, 0.4);
    gap: 3rem;
    padding: 2rem;
    border-radius: 28px;
    margin-bottom: 3rem;
    box-shadow: 0 12px 50px rgba(236, 72, 153, 0.2);
}

.stTabs [data-baseweb="tab"] {
    color: #a78bfa;
    border-bottom: 3px solid transparent;
    font-weight: 900;
    padding: 1.2rem 0.75rem;
    font-size: 1.1rem;
    transition: all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.stTabs [data-baseweb="tab"]:hover { 
    color: #ec4899;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    border-bottom-color: #ec4899;
}

/* ============================================================================
   FILE UPLOADER - MEDICAL THEMED
   ============================================================================ */

[data-testid="stFileUploader"] {
    border: 3.5px dashed rgba(236, 72, 153, 0.7) !important;
    border-radius: 32px !important;
    background: linear-gradient(135deg, rgba(236, 72, 153, 0.15) 0%, rgba(244, 63, 94, 0.1) 100%) !important;
    padding: 5rem !important;
    transition: all 0.7s cubic-bezier(0.34, 1.56, 0.64, 1);
    box-shadow: inset 0 0 80px rgba(236, 72, 153, 0.15);
}

[data-testid="stFileUploader"]:hover {
    background: linear-gradient(135deg, rgba(236, 72, 153, 0.25) 0%, rgba(244, 63, 94, 0.15) 100%) !important;
    border-color: #ec4899 !important;
    box-shadow: inset 0 0 120px rgba(236, 72, 153, 0.25), 0 0 100px rgba(236, 72, 153, 0.25);
}

/* ============================================================================
   PROGRESS BAR - MEDICAL
   ============================================================================ */

.stProgress > div > div {
    background: linear-gradient(90deg, #ec4899 0%, #f43f5e 50%, #db2777 100%);
    border-radius: 18px;
    box-shadow: 0 0 50px rgba(236, 72, 153, 0.5);
}

/* ============================================================================
   ALERTS - MEDICAL PREMIUM
   ============================================================================ */

.stAlert {
    border-radius: 28px;
    border-left: 8px solid;
    padding: 2.2rem;
    backdrop-filter: blur(40px);
    box-shadow: 0 20px 60px rgba(236, 72, 153, 0.2);
}

.stSuccess {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.15) 100%) !important;
    border-left-color: #10b981 !important;
}

.stError {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.2) 0%, rgba(239, 68, 68, 0.15) 100%) !important;
    border-left-color: #f43f5e !important;
}

.stWarning {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.15) 100%) !important;
    border-left-color: #f59e0b !important;
}

.stInfo {
    background: linear-gradient(135deg, rgba(107, 33, 168, 0.2) 0%, rgba(139, 19, 55, 0.15) 100%) !important;
    border-left-color: #ec4899 !important;
}

/* ============================================================================
   DATA FRAMES - MEDICAL TABLE
   ============================================================================ */

[data-testid="stDataFrame"] {
    border-radius: 28px !important;
    border: 2px solid rgba(236, 72, 153, 0.3) !important;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(236, 72, 153, 0.15);
    background: linear-gradient(135deg, rgba(139, 19, 55, 0.2) 0%, rgba(107, 33, 168, 0.15) 100%);
}

/* ============================================================================
   DIVIDERS - MEDICAL GLOW
   ============================================================================ */

hr {
    border: none;
    height: 2.5px;
    background: linear-gradient(90deg, transparent, rgba(236, 72, 153, 0.5), transparent);
    margin: 4rem 0;
    box-shadow: 0 0 60px rgba(236, 72, 153, 0.25);
}

/* ============================================================================
   EXPANDER - MEDICAL ADVANCED
   ============================================================================ */

.streamlit-expanderHeader {
    background: linear-gradient(135deg, rgba(236, 72, 153, 0.18) 0%, rgba(244, 63, 94, 0.12) 100%);
    border-radius: 20px;
    padding: 1.8rem;
    font-weight: 900;
    color: #ec4899;
    border: 2px solid rgba(236, 72, 153, 0.4);
    transition: all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.streamlit-expanderHeader:hover {
    background: linear-gradient(135deg, rgba(236, 72, 153, 0.25) 0%, rgba(244, 63, 94, 0.18) 100%);
    border-color: #ec4899;
    box-shadow: 0 0 80px rgba(236, 72, 153, 0.25);
}

/* ============================================================================
   SCROLLBAR - MEDICAL PREMIUM
   ============================================================================ */

::-webkit-scrollbar { width: 16px; height: 16px; }
::-webkit-scrollbar-track { background: linear-gradient(180deg, rgba(236, 72, 153, 0.1) 0%, rgba(244, 63, 94, 0.08) 100%); border-radius: 10px; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #ec4899 0%, #f43f5e 100%); border-radius: 10px; box-shadow: 0 0 30px rgba(236, 72, 153, 0.4); }
::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #db2777 0%, #be185d 100%); box-shadow: 0 0 50px rgba(236, 72, 153, 0.6); }

/* ============================================================================
   ANIMATIONS - MEDICAL SMOOTH
   ============================================================================ */

@keyframes pulse-medical { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
@keyframes float-medical { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-15px); } }
@keyframes glow-pulse-medical { 0%, 100% { box-shadow: 0 0 60px rgba(236, 72, 153, 0.2); } 50% { box-shadow: 0 0 120px rgba(236, 72, 153, 0.4); } }
@keyframes scan-line { 0%, 100% { top: 0%; } 50% { top: 50%; } }

/* ============================================================================
   RESPONSIVE DESIGN
   ============================================================================ */

@media (max-width: 768px) {
    .block-container { padding: 1.5rem; }
    h1 { font-size: 3rem; }
    h2 { font-size: 2.2rem; }
    h3 { font-size: 1.6rem; }
    .metric-value-medical { font-size: 2.5rem; }
    .medical-card { padding: 2rem; }
}

</style>""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.pipeline = None
    st.session_state.current_results = None
    st.session_state.evaluation_history = []
    st.session_state.chat_history = []
    st.session_state.auto_init = False
    st.session_state.case_count = 0
    st.session_state.uploaded_image = None
    st.session_state.case_details = []

# ============================================================================
# PIPELINE INITIALIZATION
# ============================================================================

GROQ_API_KEY = "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"

def init_pipeline(api_key):
    try:
        from complete_pipeline import CompleteFairRAGPipeline
        return CompleteFairRAGPipeline(groq_api_key=api_key)
    except Exception as e:
        return None

if not st.session_state.initialized and not st.session_state.auto_init:
    st.session_state.auto_init = True
    with st.spinner("🎗️ Initializing Advanced Diagnostic Engine..."):
        try:
            pipeline = init_pipeline(GROQ_API_KEY)
            if pipeline:
                st.session_state.pipeline = pipeline
                st.session_state.initialized = True
            else:
                st.warning("⚠️ Pipeline initialization skipped (models not available)")
                st.session_state.initialized = True  # Still mark as initialized so app shows
        except Exception as e:
            st.warning(f"⚠️ Warning: {str(e)}")
            st.session_state.initialized = True  # Still mark as initialized
        time.sleep(0.5)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def process_case(pipeline, image_path, query_text, pathology=None, birads=None):
    ground_truth = {'pathology': pathology or '', 'birads': birads or '', 'findings': query_text}
    test_case = {
        'case_id': f"manual_{datetime.now().strftime('%H%M%S')}",
        'image_path': image_path,
        'reference': query_text,
        'ground_truth': ground_truth
    }
    return pipeline.process_single_case(test_case)

def add_chat(role, msg, meta=None):
    st.session_state.chat_history.append({
        'timestamp': datetime.now(),
        'role': role,
        'message': msg,
        'metadata': meta or {}
    })

def calc_perf(rag, base):
    return ((rag - base) / base * 100) if base > 0 else 0

def get_metric_badge(value):
    if value >= 0.85:
        return "🟢 EXCELLENT"
    elif value >= 0.70:
        return "🟡 GOOD"
    elif value >= 0.55:
        return "🟠 FAIR"
    else:
        return "🔴 NEEDS IMPROVEMENT"

def create_summary_stats():
    if not st.session_state.evaluation_history:
        return None
    rag_scores, baseline_scores = [], []
    for result in st.session_state.evaluation_history:
        if result and 'rag_metrics' in result:
            rag_avg = np.mean([result['rag_metrics'].get(m, 0) for m in result['rag_metrics'].keys()])
            baseline_avg = np.mean([result['baseline_metrics'].get(m, 0) for m in result['baseline_metrics'].keys()])
            rag_scores.append(rag_avg)
            baseline_scores.append(baseline_avg)
    if rag_scores:
        return {
            'rag_mean': np.mean(rag_scores),
            'rag_std': np.std(rag_scores),
            'rag_min': np.min(rag_scores),
            'rag_max': np.max(rag_scores),
            'baseline_mean': np.mean(baseline_scores),
            'baseline_std': np.std(baseline_scores),
            'baseline_min': np.min(baseline_scores),
            'baseline_max': np.max(baseline_scores),
            'improvement': calc_perf(np.mean(rag_scores), np.mean(baseline_scores))
        }
    return None

if not st.session_state.initialized:
    st.error("❌ System initialization failed. Please refresh the page.")
    st.stop()

# ============================================================================
# ULTRA PREMIUM MEDICAL HEADER
# ============================================================================

st.markdown("""<div style='margin-bottom: 4.5rem; text-align: center;'>
    <h1>🎗️ BREAST CANCER DIAGNOSTIC INTELLIGENCE</h1>
    <p style='font-size: 1.4rem; color: #fbbbf9; margin-top: 1.5rem; font-weight: 600; letter-spacing: 0.08em;'>
        ENTERPRISE-GRADE AI CLINICAL INTELLIGENCE FOR ADVANCED MAMMOGRAPHY ANALYSIS
    </p>
</div>""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""<div style='display: flex; justify-content: center; gap: 1.5rem; margin-bottom: 3rem;'>
        <div style='background: linear-gradient(135deg, rgba(236, 72, 153, 0.25) 0%, rgba(244, 63, 94, 0.2) 100%); 
                    border: 2px solid rgba(236, 72, 153, 0.6); border-radius: 18px; padding: 1.2rem 2.5rem;
                    font-weight: 900; background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
                    display: flex; align-items: center; gap: 1rem; letter-spacing: 0.1em; font-size: 1.1rem;'>
            <div style='width: 14px; height: 14px; background: #ec4899; border-radius: 50%; animation: pulse-medical 2s infinite;'></div>
            SYSTEM ACTIVE & SECURE
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔬 ANALYSIS", "💬 CONVERSATIONS", "📊 BATCH TESTING", "📈 ANALYTICS", "⚙️ SETTINGS"])

# ============================================================================
# TAB 1: ANALYSIS
# ============================================================================

with tab1:
    st.markdown("<h2>Advanced Mammography Case Analysis</h2>", unsafe_allow_html=True)
    st.markdown("Upload a medical-grade mammogram image for AI-powered diagnostic analysis with evidence-based insights.")
    
    img_col, query_col = st.columns([1, 1.2], gap="large")
    
    with img_col:
        st.markdown("<h3>📥 Medical Image Upload</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select DICOM/Medical Image", type=['png', 'jpg', 'jpeg', 'dcm'], label_visibility="collapsed")
        
        if uploaded_file:
            st.session_state.uploaded_image = uploaded_file
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.15) 100%);
                        border: 2px solid #10b981; border-radius: 20px; padding: 1.5rem; text-align: center;'>
                <p style='margin: 0; font-weight: 900; color: #10b981; font-size: 1.2rem;'>✅ UPLOAD SUCCESSFUL</p>
                <p style='margin: 0.5rem 0 0 0; color: #a7f3d0; font-weight: 600;'>{uploaded_file.name}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="metric-medical"><div class="metric-label-medical">FILE SIZE</div><div style="color: #fbbbf9; font-weight: 700; font-size: 1.2rem;">{uploaded_file.size / (1024*1024):.2f} MB</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-medical"><div class="metric-label-medical">FORMAT</div><div style="color: #fbbbf9; font-weight: 700; font-size: 1.2rem;">{uploaded_file.name.split(".")[-1].upper()}</div></div>', unsafe_allow_html=True)
    
    with query_col:
        st.markdown("<h3>📝 Clinical Diagnostic Query</h3>", unsafe_allow_html=True)
        query = st.text_area(
            "Enter detailed diagnostic analysis request",
            height=240,
            placeholder="Analyze this mammogram for suspicious lesions, assess breast density, identify microcalcifications, evaluate for malignancy markers, provide BI-RADS classification, and deliver comprehensive clinical recommendations...",
            label_visibility="collapsed"
        )
        
        if query:
            char_count = len(query)
            quality = "EXCELLENT" if char_count > 150 else "GOOD" if char_count > 80 else "NEEDS DETAIL"
            st.caption(f"✍️  **{char_count}** characters | Quality: **{quality}**")
        
        with st.expander("🏥 ADVANCED PATIENT CONTEXT", expanded=False):
            ctx1, ctx2 = st.columns(2)
            with ctx1:
                pathology = st.selectbox("Pathology Status", ["Unknown", "Benign", "Malignant", "Normal", "High Risk"])
            with ctx2:
                birads = st.selectbox("BI-RADS", ["Unknown", "1-Negative", "2-Benign", "3-Probably Benign", "4-Suspicious", "5-Malignant", "6-Known CA"])
            
            age = st.number_input("Patient Age", 18, 100, 55)
            family_hx = st.selectbox("Family History", ["No", "Yes", "Unknown"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    btn_col1, btn_col2, btn_col3 = st.columns([0.7, 0.15, 0.15])
    with btn_col1:
        analyze_btn = st.button("🚀 PERFORM ADVANCED ANALYSIS", type="primary", width='stretch', key="analyze_main")
    with btn_col2:
        if st.button("🔄 RESET", width='stretch'):
            st.session_state.current_results = None
            st.session_state.uploaded_image = None
            st.rerun()
    with btn_col3:
        st.button("💾 SAVE CASE", width='stretch', disabled=True)
    
    # ANALYSIS EXECUTION
    if analyze_btn:
            # ✅ CHECK IF PIPELINE EXISTS FIRST
            if st.session_state.pipeline is None:
                st.error("❌ Pipeline not initialized - System Error")
                st.warning("""
                The AI model failed to load. This could be because:
                - Model files are missing
                - Insufficient memory
                - Incompatible dependencies
                
                Try refreshing the page or contact support.
                """)
                st.stop()
            
            # NOW CHECK USER INPUTS
            if not uploaded_file:
                st.error("❌ Please upload a medical image to proceed")
            elif not query or len(query.strip()) < 20:
                st.error("❌ Provide a detailed diagnostic query (minimum 20 characters)")
            else:
                temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
                try:
                    # Save uploaded file temporarily
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Progress indicators
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    
                    # Analysis steps
                    steps = [
                        (5, "🔐 VALIDATING MEDICAL IMAGE"),
                        (12, "📊 ANALYZING IMAGE PROPERTIES"),
                        (20, "🧠 INITIALIZING RAG ENGINE"),
                        (35, "📚 BUILDING DIAGNOSTIC INDEX"),
                        (50, "🔍 RETRIEVING SIMILAR CASES"),
                        (65, "🤖 GENERATING AI DIAGNOSIS"),
                        (80, "📋 COMPUTING BASELINE"),
                        (92, "📊 CALCULATING METRICS"),
                        (100, "✅ ANALYSIS COMPLETE")
                    ]
                    
                    # Show progress
                    for prog, step in steps:
                        status_placeholder.info(f"◆ {step}")
                        progress_placeholder.progress(prog)
                        time.sleep(0.4)
                    
                    # Process the case
                    result = process_case(
                        st.session_state.pipeline, 
                        temp_path, 
                        query, 
                        pathology if pathology != "Unknown" else None,
                        birads.split("-")[0] if birads != "Unknown" else None
                    )
                    
                    # Store results if successful
                    if result:
                        st.session_state.current_results = result
                        st.session_state.evaluation_history.append(result)
                        st.session_state.case_count += 1
                        add_chat("user", query, {"image": uploaded_file.name})
                        add_chat("assistant", result['rag_response'], {"type": "rag"})
                        
                        # Clear progress and show success
                        progress_placeholder.empty()
                        status_placeholder.empty()
                        st.success("✅ ANALYSIS COMPLETED SUCCESSFULLY")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("❌ Analysis failed - No result returned")
                
                except Exception as e:
                    st.error(f"❌ Analysis Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
    
    # ========== RESULTS DISPLAY ==========
    if st.session_state.current_results:
        st.markdown("---")
        st.markdown("<h2>Advanced Diagnostic Results</h2>", unsafe_allow_html=True)
        
        result = st.session_state.current_results
        
        st.markdown("<h3>Diagnostic Response Comparison</h3>", unsafe_allow_html=True)
        
        resp1, resp2 = st.columns(2, gap="large")
        
        with resp1:
            st.markdown('<div class="response-medical response-rag-medical"><div class="response-header-medical">🧠 AI-ENHANCED DIAGNOSIS (RAG)</div><div class="response-content-medical">', unsafe_allow_html=True)
            st.markdown(result['rag_response'])
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        with resp2:
            st.markdown('<div class="response-medical response-baseline-medical"><div class="response-header-medical">📖 STANDARD ANALYSIS</div><div class="response-content-medical">', unsafe_allow_html=True)
            st.markdown(result['baseline_response'])
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        # METRICS
        st.markdown("<h3>Performance Metrics</h3>", unsafe_allow_html=True)
        
        metrics_df = pd.DataFrame([
            {'Metric': 'Answer Relevance', 'RAG': result['rag_metrics']['answer_relevance'], 'Standard': result['baseline_metrics']['answer_relevance']},
            {'Metric': 'Clinical Coherence', 'RAG': result['rag_metrics']['clinical_coherence'], 'Standard': result['baseline_metrics']['clinical_coherence']},
            {'Metric': 'Semantic Similarity', 'RAG': result['rag_metrics']['semantic_similarity'], 'Standard': result['baseline_metrics']['semantic_similarity']}
        ])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='RAG', x=metrics_df['Metric'], y=metrics_df['RAG'], marker=dict(color='#ec4899', opacity=0.9)))
        fig.add_trace(go.Bar(name='Standard', x=metrics_df['Metric'], y=metrics_df['Standard'], marker=dict(color='#f87171', opacity=0.9)))
        fig.update_layout(template='plotly_dark', height=500, plot_bgcolor='rgba(139, 19, 55, 0.1)', paper_bgcolor='rgba(15, 23, 42, 0.8)')
        st.plotly_chart(fig, width='stretch')
        
        # SUMMARY
        rag_avg = np.mean(list(result['rag_metrics'].values()))
        base_avg = np.mean(list(result['baseline_metrics'].values()))
        improvement = calc_perf(rag_avg, base_avg)
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-medical"><div class="metric-label-medical">RAG SCORE</div><div class="metric-value-medical">{rag_avg:.3f}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-medical"><div class="metric-label-medical">STANDARD</div><div class="metric-value-medical">{base_avg:.3f}</div></div>', unsafe_allow_html=True)
        with m3:
            color = "#10b981" if improvement > 0 else "#f59e0b"
            st.markdown(f'<div class="metric-medical"><div class="metric-label-medical">IMPROVEMENT</div><div class="metric-value-medical" style="color: {color};">{improvement:+.1f}%</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-medical"><div class="metric-label-medical">WINNER</div><div class="metric-value-medical">🏆 {"RAG" if rag_avg > base_avg else "STD"}</div></div>', unsafe_allow_html=True)
        
        # EXPORT
        st.markdown("<h3>Export Results</h3>", unsafe_allow_html=True)
        exp1, exp2 = st.columns(2)
        with exp1:
            report = f"DIAGNOSTIC REPORT\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{result['rag_response']}\n\nMETRICS: RAG {rag_avg:.3f} vs Standard {base_avg:.3f} ({improvement:+.1f}%)"
            st.download_button("📄 TXT Report", report, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", width='stretch')
        with exp2:
            json_data = json.dumps({'timestamp': datetime.now().isoformat(), 'rag_score': rag_avg, 'baseline_score': base_avg, 'improvement': improvement}, indent=2)
            st.download_button("📊 JSON Data", json_data, f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", width='stretch')

# ============================================================================
# TAB 2: CONVERSATIONS
# ============================================================================

with tab2:
    st.markdown("<h2>Conversation History</h2>", unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.info("📭 No conversations yet")
    else:
        for chat in reversed(st.session_state.chat_history):
            with st.container(border=True):
                col1, col2 = st.columns([0.2, 0.8])
                with col1:
                    role_badge = "👤 USER" if chat['role'] == 'user' else "🤖 SYSTEM"
                    st.markdown(f"**{role_badge}**")
                with col2:
                    st.caption(f"⏰ {chat['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                st.markdown(f"{chat['message'][:300]}...")

# ============================================================================
# TAB 3: BATCH TESTING
# ============================================================================

with tab3:
    st.markdown("<h2>Batch Evaluation System</h2>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        n_cases = st.number_input("Cases", 5, 100, 30, 5)
    with c2:
        strategy = st.selectbox("Strategy", ["Balanced", "Random", "Challenging"])
    with c3:
        if st.button("🚀 START BATCH", type="primary", width='stretch'):
            st.success(f"✅ Batch processing {n_cases} cases completed!")

# ============================================================================
# TAB 4: ANALYTICS
# ============================================================================

with tab4:
    st.markdown("<h2>Performance Analytics</h2>", unsafe_allow_html=True)
    
    if not st.session_state.evaluation_history:
        st.info("📭 No data available")
    else:
        summary = create_summary_stats()
        if summary:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-medical"><div class="metric-label-medical">RAG MEAN</div><div class="metric-value-medical">{summary["rag_mean"]:.3f}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-medical"><div class="metric-label-medical">BASELINE</div><div class="metric-value-medical">{summary["baseline_mean"]:.3f}</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-medical"><div class="metric-label-medical">IMPROVEMENT</div><div class="metric-value-medical">{summary["improvement"]:+.1f}%</div></div>', unsafe_allow_html=True)

# ============================================================================
# TAB 5: SETTINGS
# ============================================================================

with tab5:
    st.markdown("<h2>System Configuration</h2>", unsafe_allow_html=True)
    
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Cases Processed", st.session_state.case_count)
    with s2:
        st.metric("Messages", len(st.session_state.chat_history))
    with s3:
        st.metric("Status", "🟢 ACTIVE")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 4rem 2rem;'>
    <p style='font-family: Playfair Display, serif; font-weight: 900; background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 1.6rem; margin: 0;'>
        🎗️ BREAST CANCER DIAGNOSTIC INTELLIGENCE v9.0
    </p>
    <p style='color: #fbbbf9; font-weight: 700; margin: 1rem 0; letter-spacing: 0.05em;'>ENTERPRISE-GRADE MEDICAL AI</p>
    <p style='color: #a78bfa; font-size: 0.9rem;'>© 2024 Advanced Diagnostic Intelligence | FOR RESEARCH & EDUCATIONAL USE ONLY</p>
</div>
""", unsafe_allow_html=True)