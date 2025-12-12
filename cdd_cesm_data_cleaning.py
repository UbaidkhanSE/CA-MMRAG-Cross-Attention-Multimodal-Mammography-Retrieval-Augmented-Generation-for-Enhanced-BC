import pandas as pd
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import torchvision.transforms as T

# Load dataset
cdd_cesm = r"C:\mammography_gpt\Dataset\complete_multimodal_dataset.csv"

df = pd.read_csv(cdd_cesm)
print(f"Dataset shape: {df.shape}")
df = df.drop(columns=['Unnamed: 11', 'Unnamed: 12'])

#print(f"Columns: {df.columns.tolist()}")

df['Image_Type'] = df['Image_Type'].replace({
    'DM': 'Digital Mammography',
    'CESM': 'Contrast-Enhanced Spectral Mammography'
}).str.lower()
#print(df['Image_Type'].head(5))

df['BIRADS'] = "birads score " + df['BIRADS'].astype(str)
#print(df[['BIRADS']].head(5))

df['Breast_Density'] = df['Breast_Density'].replace({
    'A': 'almost entirely fatty',
    'B': 'scattered areas of fibroglandular density',
    'C': 'heterogeneously dense',
    'D': 'extremely dense'
}).str.lower()

# Optional: add a prefix for clarity
df['Breast_Density'] = "breast density  " + df['Breast_Density']
df['Side'] = df['Side'].str.lower()
df['View'] = df['View'].str.lower()
df['Age'] = "age " + df['Age'].astype(str)
df['Pathology'] = df['Pathology'].str.lower()
df['Findings'] = df['Findings'].str.lower()
df['Findings'] = df['Findings'].str.strip()


#print(list(df.columns))
import re
import pandas as pd

def clean_text_column(series):
    """
    Clean a string column for embeddings:
    - lowercase
    - remove repeated words
    - strip extra spaces
    """
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()  # lowercase
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)  # remove repeated words
        text = re.sub(r'\s+', ' ', text)  # normalize spaces
        return text.strip()
    
    return series.apply(clean_text)

text_cols = [
    *[f'Question_{i}' for i in range(1, 11)],
    *[f'Generated_Answer_{i}' for i in range(1, 11)],
    'Generated_Caption',
    'Clinical_Reports'
]

for col in text_cols:
    if col in df.columns:
        df[col] = clean_text_column(df[col])


# Check shape, data types, and missing values
print(df.info())
print(df.isna().sum())

# Preview first 5 rows
print(df.head())

# Identify all string columns
string_cols = df.select_dtypes(include='object').columns.tolist()

for col in string_cols:
    print(f"Column: {col}")
    print("Sample values:", df[col].head(5).tolist())
    print("Unique values count:", df[col].nunique())
    print("-"*50)

# Check if any column still has uppercase letters
for col in string_cols:
    if df[col].str.contains(r'[A-Z]').any():
        print(f"Uppercase letters found in {col}")

# Check if any column has repeated words
for col in string_cols:
    if df[col].str.contains(r'\b(\w+)( \1\b)+').any():
        print(f"Repeated words found in {col}")

################ image prerocessing #######################3333
# Define target size for CLIP
TARGET_SIZE = (224, 224)

# Function to resize and save images in-place
def resize_image_inplace(img_path, target_size=TARGET_SIZE):
    if os.path.exists(img_path):
        with Image.open(img_path) as img:
            img = img.convert("RGB")            # ensure RGB
            img = img.resize(target_size)       # resize
            img.save(img_path)                  # overwrite same path
    else:
        print(f"File not found: {img_path}")

# Apply to all images in your dataset
for path in df['Image_Path']:
    resize_image_inplace(path)

print("All images are resized and ready for CLIP.")

df.to_csv(cdd_cesm, index=False)
print(f"Clean dataset saved to {cdd_cesm}")