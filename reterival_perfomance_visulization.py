import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Data from your table
data = {
    'Model': [
        'Baseline1_ImageOnly', 'Baseline2_TextOnly', 'Baseline3_SimpleFusion', 'Baseline4_AttentionFusion',
        'PatchBaseline1_ImageOnlyPatches', 'PatchBaseline2_TextOnlyPatches', 'PatchBaseline3_SimpleFusionPatches',
        'PatchBaseline4_AttentionFusionPatches', 'PatchBaseline5_DeepFusionPatches', 'Proposed_CrossAttention'
    ],
    'P@1': [0.4550, 0.5500, 0.7200, 0.7050, 0.4450, 0.5900, 0.8900, 0.9350, 0.8950, 0.9950],
    'P@5': [0.4110, 0.4840, 0.6700, 0.6820, 0.4250, 0.5410, 0.8760, 0.9200, 0.8870, 0.9940],
    'P@10': [0.4075, 0.4590, 0.6390, 0.6745, 0.4020, 0.5005, 0.8745, 0.9170, 0.8740, 0.9920],
    'P@100': [0.3553, 0.3850, 0.5486, 0.5975, 0.3454, 0.4009, 0.7531, 0.7888, 0.7235, 0.8458],
    'mAP': [0.4040, 0.4441, 0.6213, 0.6584, 0.3992, 0.4768, 0.8642, 0.9158, 0.8454, 0.9914],
    'Category': ['WITHOUT PATCHES', 'WITHOUT PATCHES', 'WITHOUT PATCHES', 'WITHOUT PATCHES',
                 'WITH PATCHES', 'WITH PATCHES', 'WITH PATCHES', 'WITH PATCHES', 'WITH PATCHES', 'PROPOSED METHOD']
}

df = pd.DataFrame(data)

# Create abstract performance landscape visualization
def create_abstract_performance_landscape():
    # Create subplots: 2x2 grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Evolution', 'Top 5 Model Ranking', 
                       'Performance Improvement Matrix', 'Model Performance Landscape'),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "heatmap"}, {"type": "scatter"}]]
    )
    
    # 1. TOP LEFT: Performance Evolution (Abstract Line Graph)
    metrics = ['P@1', 'P@5', 'P@10', 'P@100', 'mAP']
    colors_evolution = ['#FF6B9D', '#C44569', '#F8B500', '#6C7B7F', '#40E0D0']
    
    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=df[metric],
                mode='lines+markers',
                name=metric,
                line=dict(color=colors_evolution[i], width=3),
                marker=dict(size=8),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 2. TOP RIGHT: Model Performance Ranking (Abstract Bars)
    top_5_models = df.nlargest(5, 'mAP')
    colors_ranking = ['gold' if cat == 'PROPOSED METHOD' else 
                     'lightcoral' if cat == 'WITH PATCHES' else 'lightblue' 
                     for cat in top_5_models['Category']]
    
    fig.add_trace(
        go.Bar(
            x=[f"Rank {i+1}" for i in range(len(top_5_models))],
            y=top_5_models['mAP'].values,
            marker_color=colors_ranking,
            name='Top 5 Models',
            text=[f'{val:.3f}' for val in top_5_models['mAP'].values],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. BOTTOM LEFT: Performance Improvement Matrix (Abstract Heatmap)
    # Create improvement matrix: how much each model improves over baseline
    baseline_scores = df.iloc[0][metrics].values  # First baseline
    improvement_matrix = []
    
    for _, model in df.iterrows():
        model_scores = model[metrics].values
        improvements = ((model_scores - baseline_scores) / baseline_scores * 100)
        improvement_matrix.append(improvements)
    
    improvement_matrix = np.array(improvement_matrix)
    
    fig.add_trace(
        go.Heatmap(
            z=improvement_matrix,
            x=metrics,
            y=[f"M{i+1}" for i in range(len(df))],  # M1, M2, etc. for models
            colorscale='RdYlGn',
            name='Improvement %',
            showscale=True,
            colorbar=dict(title="Improvement %")
        ),
        row=2, col=1
    )
    
    # 4. BOTTOM RIGHT: Proposed Method Dominance (Abstract Bubble Chart)
    # Create abstract representation where bubble size = overall performance
    overall_performance = df[metrics].mean(axis=1)
    
    fig.add_trace(
        go.Scatter(
            x=df['mAP'],
            y=overall_performance,
            mode='markers',
            marker=dict(
                size=[40 if cat == 'PROPOSED METHOD' else 20 for cat in df['Category']],
                color=['gold' if cat == 'PROPOSED METHOD' else 
                       'lightcoral' if cat == 'WITH PATCHES' else 'lightblue' 
                       for cat in df['Category']],
                opacity=0.8,
                line=dict(width=2, color='black')
            ),
            text=df['Model'],
            textposition='top center',
            name='Model Performance',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout for modern journal paper look
    fig.update_layout(
        title={
            'text': 'Model Performance Analysis - Abstract Representation',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'family': 'Arial Black', 'color': '#2c3e50'}
        },
        height=800,
        width=1400,
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='white',
        font=dict(size=12, family='Arial'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Model Index", row=1, col=1)
    fig.update_yaxes(title_text="Performance Score", row=1, col=1)
    
    fig.update_xaxes(title_text="Top Models", row=1, col=2)
    fig.update_yaxes(title_text="mAP Score", row=1, col=2)
    
    fig.update_xaxes(title_text="Metrics", row=2, col=1)
    fig.update_yaxes(title_text="Models", row=2, col=1)
    
    fig.update_xaxes(title_text="mAP Score", row=2, col=2)
    fig.update_yaxes(title_text="Overall Performance", row=2, col=2)
    
    return fig

# Create the visualization
def main():
    print("🎨 Creating abstract model performance visualization...")
    
    # Generate the figure
    fig = create_abstract_performance_landscape()
    
    # Save to project directory (no browser opening)
    output_file = "model_performance_abstract.html"
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"✅ Interactive visualization saved: {output_file}")
    
    # Save high-quality PNG for journal paper
    try:
        png_file = "model_performance_abstract.png"
        fig.write_image(png_file, width=1400, height=800, scale=2)
        print(f"✅ High-quality PNG saved: {png_file}")
    except Exception as e:
        print(f"⚠️  PNG save failed (install kaleido: pip install kaleido)")
        print(f"   Error: {e}")
    
    # Print insights
    best_model = df.loc[df['mAP'].idxmax()]
    print(f"\n🏆 Top Performer: {best_model['Model']}")
    print(f"   mAP Score: {best_model['mAP']:.4f}")
    print(f"   Category: {best_model['Category']}")
    
    improvement = (best_model['mAP'] - df['mAP'].min()) / df['mAP'].min() * 100
    print(f"📈 Performance Improvement: {improvement:.1f}% over worst model")
    
    print(f"\n💾 Files saved in current directory:")
    print(f"   - {output_file} (interactive)")
    print(f"   - {png_file} (journal paper)")

if __name__ == "__main__":
    main()