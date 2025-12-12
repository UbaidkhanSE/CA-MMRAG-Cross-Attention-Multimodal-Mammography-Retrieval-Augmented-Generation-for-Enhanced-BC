import PyPDF2
import pandas as pd
import numpy as np
import json
import faiss
import os
from pathlib import Path
from tqdm import tqdm
import open_clip
import torch
import torch.nn.functional as F

class Literature768Embedder:
    def __init__(self, literature_path=r"C:\mammography_gpt\Literature"):
        self.literature_path = Path(literature_path)
        self.storage_path = Path("mammography_retrieval_storage")
        
        # Initialize BiomedCLIP for consistency with your training
        self.model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.tokenizer = open_clip.get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Create literature storage
        os.makedirs(self.storage_path / "literature", exist_ok=True)
        
        print(f"Literature embedder initialized on {self.device}")
        print("Will save embeddings as 768-dimensional to match clinical cases")
    
    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF with error handling"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + " "
                return text.strip()
        except Exception as e:
            print(f"Error reading {pdf_path.name}: {e}")
            return ""
    
    def chunk_text(self, text, max_tokens=200, overlap=50):
        """Split text into overlapping chunks for better coverage"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_tokens - overlap):
            chunk = " ".join(words[i:i + max_tokens])
            if len(chunk.strip()) > 50:  # Skip very short chunks
                chunks.append(chunk.strip())
        
        return chunks
    
    def encode_text_chunks_768(self, chunks, batch_size=32):
        """Encode text chunks using BiomedCLIP and pad to 768 dimensions"""
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                tokens = self.tokenizer(batch_chunks).to(self.device)
                chunk_embeddings = self.model.encode_text(tokens)
                
                # BiomedCLIP text encoder gives 512-dim, pad to 768 to match clinical cases
                if chunk_embeddings.shape[1] == 512:
                    padding = torch.zeros(chunk_embeddings.shape[0], 256).to(self.device)
                    chunk_embeddings = torch.cat([chunk_embeddings, padding], dim=1)
                
                chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)
                embeddings.append(chunk_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def process_literature_768(self):
        """Process all PDFs and create 768-dimensional embeddings"""
        print("Processing literature PDFs to 768-dimensional embeddings...")
        
        all_embeddings = []
        all_metadata = []
        chunk_id = 0
        
        pdf_files = list(self.literature_path.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            # Extract text
            full_text = self.extract_pdf_text(pdf_file)
            if not full_text:
                continue
                
            # Create chunks
            chunks = self.chunk_text(full_text)
            if not chunks:
                continue
            
            # Encode chunks to 768 dimensions
            chunk_embeddings = self.encode_text_chunks_768(chunks)
            all_embeddings.append(chunk_embeddings)
            
            # Create metadata for each chunk
            for i, chunk in enumerate(chunks):
                metadata = {
                    'chunk_id': chunk_id,
                    'paper_filename': pdf_file.name,
                    'paper_title': pdf_file.stem.replace('_', ' ').title(),
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'text_content': chunk[:500],  # Store first 500 chars
                    'full_text_length': len(full_text),
                    'chunk_length': len(chunk)
                }
                all_metadata.append(metadata)
                chunk_id += 1
        
        # Combine all embeddings
        complete_literature_embeddings = np.vstack(all_embeddings)
        
        print(f"Created {complete_literature_embeddings.shape[0]} text chunks from {len(pdf_files)} papers")
        print(f"Literature embeddings shape: {complete_literature_embeddings.shape} (768-dimensional)")
        
        # Verify 768 dimensions
        if complete_literature_embeddings.shape[1] != 768:
            raise ValueError(f"Expected 768 dimensions, got {complete_literature_embeddings.shape[1]}")
        
        # Save 768-dimensional embeddings
        np.save(self.storage_path / "literature" / "literature_768_embeddings.npy", 
                complete_literature_embeddings)
        
        # Save metadata
        with open(self.storage_path / "literature" / "literature_768_metadata.json", 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        # Create FAISS index for 768-dimensional embeddings
        index = faiss.IndexFlatIP(768)
        index.add(complete_literature_embeddings.astype('float32'))
        
        faiss.write_index(index, str(self.storage_path / "literature" / "literature_768_index.index"))
        
        # Save literature info
        lit_info = {
            'total_papers': len(pdf_files),
            'total_chunks': len(all_metadata),
            'embedding_dimension': 768,
            'original_biomedclip_dim': 512,
            'padded_to_match_clinical': True,
            'average_chunks_per_paper': len(all_metadata) / len(pdf_files) if pdf_files else 0,
            'storage_path': str(self.storage_path / "literature")
        }
        
        with open(self.storage_path / "literature" / "literature_768_info.json", 'w') as f:
            json.dump(lit_info, f, indent=2)
        
        print(f"768-dimensional literature embeddings saved: {complete_literature_embeddings.shape}")
        return complete_literature_embeddings, all_metadata
    
    def test_literature_768_search(self, query="mammography AI screening"):
        """Test literature search with 768-dimensional embeddings"""
        try:
            # Load 768-dimensional components
            embeddings = np.load(self.storage_path / "literature" / "literature_768_embeddings.npy")
            index = faiss.read_index(str(self.storage_path / "literature" / "literature_768_index.index"))
            
            with open(self.storage_path / "literature" / "literature_768_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            print(f"Loaded literature embeddings: {embeddings.shape}")
            
            # Encode query to 768 dimensions
            with torch.no_grad():
                query_tokens = self.tokenizer([query]).to(self.device)
                query_embedding = self.model.encode_text(query_tokens)
                
                # Pad query to 768 dimensions
                if query_embedding.shape[1] == 512:
                    padding = torch.zeros(1, 256).to(self.device)
                    query_embedding = torch.cat([query_embedding, padding], dim=1)
                
                query_embedding = F.normalize(query_embedding, p=2, dim=1)
            
            # Search
            scores, indices = index.search(query_embedding.cpu().numpy(), k=5)
            
            print(f"\nLiterature 768-dim search test: '{query}'")
            for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
                result = metadata[idx]
                print(f"{i+1}. {result['paper_title']} (Score: {score:.4f})")
                print(f"   {result['text_content'][:100]}...")
            
            return True
            
        except Exception as e:
            print(f"Literature 768-dim search test failed: {e}")
            return False
    
    def integrate_with_clinical_storage(self):
        """Integrate 768-dim literature with existing clinical storage"""
        try:
            # Load clinical embeddings (should be 768-dim)
            clinical_embeddings = np.load(self.storage_path / "embeddings" / "complete_dataset_embeddings.npy")
            with open(self.storage_path / "metadata" / "complete_dataset_metadata.json", 'r') as f:
                clinical_metadata = json.load(f)
            
            # Load 768-dim literature embeddings
            literature_embeddings = np.load(self.storage_path / "literature" / "literature_768_embeddings.npy")
            with open(self.storage_path / "literature" / "literature_768_metadata.json", 'r') as f:
                literature_metadata = json.load(f)
            
            print(f"Clinical embeddings: {clinical_embeddings.shape}")
            print(f"Literature embeddings: {literature_embeddings.shape}")
            
            # Verify both are 768-dimensional
            if clinical_embeddings.shape[1] != 768:
                raise ValueError(f"Clinical embeddings are {clinical_embeddings.shape[1]}-dim, expected 768")
            if literature_embeddings.shape[1] != 768:
                raise ValueError(f"Literature embeddings are {literature_embeddings.shape[1]}-dim, expected 768")
            
            # Combine embeddings
            unified_embeddings = np.vstack([clinical_embeddings, literature_embeddings])
            
            # Prepare unified metadata
            unified_metadata = []
            
            # Add clinical metadata
            for item in clinical_metadata:
                item['source_type'] = 'clinical_case'
                item['unified_index'] = len(unified_metadata)
                unified_metadata.append(item)
            
            # Add literature metadata
            for item in literature_metadata:
                item['source_type'] = 'literature'
                item['unified_index'] = len(unified_metadata)
                unified_metadata.append(item)
            
            # Normalize unified embeddings
            unified_embeddings = unified_embeddings / np.linalg.norm(unified_embeddings, axis=1, keepdims=True)
            
            # Save unified 768-dimensional database
            np.save(self.storage_path / "embeddings" / "unified_768_embeddings.npy", unified_embeddings)
            
            with open(self.storage_path / "metadata" / "unified_768_metadata.json", 'w') as f:
                json.dump(unified_metadata, f, indent=2)
            
            # Create unified FAISS index
            unified_index = faiss.IndexFlatIP(768)
            unified_index.add(unified_embeddings.astype('float32'))
            faiss.write_index(unified_index, str(self.storage_path / "indices" / "unified_768_index.index"))
            
            # Save integration info
            integration_info = {
                'total_vectors': unified_embeddings.shape[0],
                'dimension': 768,
                'clinical_cases': len(clinical_metadata),
                'literature_chunks': len(literature_metadata),
                'storage_files': {
                    'embeddings': 'unified_768_embeddings.npy',
                    'metadata': 'unified_768_metadata.json',
                    'index': 'unified_768_index.index'
                }
            }
            
            with open(self.storage_path / "metadata" / "unified_768_info.json", 'w') as f:
                json.dump(integration_info, f, indent=2)
            
            print("SUCCESS: 768-dimensional unified database created!")
            print(f"Total vectors: {unified_embeddings.shape[0]}")
            print(f"Clinical cases: {len(clinical_metadata)}")
            print(f"Literature chunks: {len(literature_metadata)}")
            print(f"Dimension: 768 (consistent)")
            
            return unified_embeddings, unified_metadata
            
        except Exception as e:
            print(f"Integration failed: {e}")
            return None, None

def main():
    embedder = Literature768Embedder()
    
    # Process literature to 768 dimensions
    embeddings, metadata = embedder.process_literature_768()
    
    # Test 768-dimensional search
    embedder.test_literature_768_search()
    
    # Integrate with clinical storage
    unified_embeddings, unified_metadata = embedder.integrate_with_clinical_storage()
    
    if unified_embeddings is not None:
        print("\n" + "="*60)
        print("LITERATURE 768-DIM INTEGRATION COMPLETE")
        print("="*60)
        print("Ready for unified retrieval: Clinical cases + Literature")
        print("Both sources now in consistent 768-dimensional space")
        print("Compatible with your cross-attention retrieval system")
    
    return embedder

if __name__ == "__main__":
    main()