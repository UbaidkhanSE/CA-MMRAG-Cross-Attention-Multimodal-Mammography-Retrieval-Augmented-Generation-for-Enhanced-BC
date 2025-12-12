"""Mammography image captions were generated using the BLIP (Bootstrapping Language-Image Pre-training) base model, which processed each image with structured metadata prompts containing clinical information including BI-RADS assessment, breast density, pathology results, and findings. The generated captions were evaluated against original clinical reports using cosine similarity scores computed through SentenceTransformer embeddings to measure semantic alignment and assess caption quality."""
"""Batch Average Similarity Score: 0.9058
📈 Overall Average Similarity Score: 0.9027"""
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch
import os
import json

# === Configuration ===
BATCH_SIZE = 2006
CSV_PATH = r"C:\Mammo_GPT\final_clean_dataset.csv"
PROGRESS_FILE = "caption_progress.json"
OUTPUT_FILE = "generated_captions.csv"

# === Load Dataset ===
df = pd.read_csv(CSV_PATH)

# === Progress Tracking ===
def load_progress():
    """Load progress from file or initialize"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"last_processed_index": -1, "total_processed": 0}

def save_progress(progress_data):
    """Save progress to file"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress_data, f)

def load_existing_results():
    """Load existing results if available"""
    if os.path.exists(OUTPUT_FILE):
        return pd.read_csv(OUTPUT_FILE)
    return pd.DataFrame()

# === Build Prompt from Metadata ===
def build_prompt(row):
    return (
        f"This is a {row['Modality']} image of the {row['Side']} breast in {row['View']} view. "
        f"BI-RADS assessment: {row['BIRADS_Description']}. "
        f"Breast density is described as {row['Breast_Density_Description']}. "
        f"Findings include: {row['Findings_Clean']}. "
        f"Pathology result: {row['Pathology']}. "
        f"Additional tags: {row['Tags']}. "
        f"Clinical report summary: {row['Report_Text_Clean']}"
    )

# === Load Models ===
print("🔄 Loading models...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
sim_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Models loaded successfully")

# === Generate Caption Function ===
def generate_caption(img_path, prompt):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"[Error generating caption: {e}]"

def calculate_similarity(ref_text, gen_caption):
    try:
        embeddings = sim_model.encode([ref_text, gen_caption], convert_to_tensor=True)
        return util.cos_sim(embeddings[0], embeddings[1]).item()
    except:
        return 0.0

# === Main Processing Logic ===
def main():
    # Load progress and existing results
    progress = load_progress()
    existing_results = load_existing_results()
    
    start_idx = progress["last_processed_index"] + 1
    end_idx = min(start_idx + BATCH_SIZE, len(df))
    
    print(f"📊 Processing batch: {start_idx + 1} to {end_idx} (Total dataset size: {len(df)})")
    
    if start_idx >= len(df):
        print("✅ All data has been processed!")
        return
    
    # Get current batch
    current_batch = df.iloc[start_idx:end_idx].copy()
    
    # Build prompts
    current_batch["Prompt"] = current_batch.apply(build_prompt, axis=1)
    
    # Process batch
    results = []
    for i, (idx, row) in enumerate(current_batch.iterrows()):
        print(f"🔄 Processing {i+1}/{len(current_batch)}: {row['Image_Path']}")
        
        # Generate caption
        caption = generate_caption(row["Image_Path"], row["Prompt"])
        
        # Calculate similarity
        similarity = calculate_similarity(row["Report_Text_Clean"], caption)
        
        # Store result with all columns
        result = {
            'Patient_ID': row['Patient_ID'],
            'Image_ID': row['Image_ID'],
            'Modality': row['Modality'],
            'Image_Path': row['Image_Path'],
            'Image_Extension': row['Image_Extension'],
            'Report_Path': row['Report_Path'],
            'Report_Text': row['Report_Text'],
            'Report_Available': row['Report_Available'],
            'Side': row['Side'],
            'View': row['View'],
            'Age': row['Age'],
            'BIRADS': row['BIRADS'],
            'Breast_Density': row['Breast_Density'],
            'Findings': row['Findings'],
            'Tags': row['Tags'],
            'Pathology': row['Pathology'],
            'Pathology_Label': row['Pathology_Label'],
            'Record_Index': row['Record_Index'],
            'Processing_Timestamp': row['Processing_Timestamp'],
            'BIRADS_Description': row['BIRADS_Description'],
            'Breast_Density_Description': row['Breast_Density_Description'],
            'Findings_Clean': row['Findings_Clean'],
            'Report_Text_Clean': row['Report_Text_Clean'],
            'Prompt': row['Prompt'],
            'Generated_Caption': caption,
            'Similarity_Score': similarity
        }
        results.append(result)
    
    # Convert to DataFrame
    batch_df = pd.DataFrame(results)
    
    # Combine with existing results
    if not existing_results.empty:
        combined_df = pd.concat([existing_results, batch_df], ignore_index=True)
    else:
        combined_df = batch_df
    
    # Save results
    combined_df.to_csv(OUTPUT_FILE, index=False)
    
    # Update progress
    progress["last_processed_index"] = end_idx - 1
    progress["total_processed"] = len(combined_df)
    save_progress(progress)
    
    # Display results
    print("\n📝 Current Batch Results:")
    print(batch_df[["Image_Path", "Generated_Caption", "Similarity_Score"]].to_string(index=False))
    
    batch_avg_score = batch_df["Similarity_Score"].mean()
    overall_avg_score = combined_df["Similarity_Score"].mean()
    
    print(f"\n📈 Batch Average Similarity Score: {batch_avg_score:.4f}")
    print(f"📈 Overall Average Similarity Score: {overall_avg_score:.4f}")
    print(f"✅ Progress: {len(combined_df)}/{len(df)} samples processed")
    print(f"💾 Results saved to: {OUTPUT_FILE}")
    print(f"📊 Progress saved to: {PROGRESS_FILE}")
    
    if end_idx < len(df):
        print(f"🔄 Run again to process next batch ({end_idx + 1} to {min(end_idx + BATCH_SIZE, len(df))})")
    else:
        print("🎉 All data processing completed!")

if __name__ == "__main__":
    main()