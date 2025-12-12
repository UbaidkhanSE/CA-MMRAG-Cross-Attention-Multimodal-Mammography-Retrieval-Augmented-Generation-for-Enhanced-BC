import numpy as np
import faiss
import json
import torch
import os
from pathlib import Path

def test_complete_storage(storage_path="mammography_retrieval_storage"):
    """Comprehensive test of all storage components including literature"""
    
    print("Testing Mammography Retrieval Storage + Literature")
    print("=" * 50)
    
    storage_path = Path(storage_path)
    passed_tests = 0
    total_tests = 0
    
    # Test 1: Folder structure (including literature)
    print("1. Testing folder structure...")
    total_tests += 1
    required_folders = ['embeddings', 'models', 'indices', 'metadata', 'literature']
    if all((storage_path / folder).exists() for folder in required_folders):
        print("   ✅ All folders exist (including literature)")
        passed_tests += 1
    else:
        missing = [f for f in required_folders if not (storage_path / f).exists()]
        print(f"   ❌ Missing folders: {missing}")
        return False
    
    # Test 2: Clinical embeddings
    print("2. Testing clinical embeddings...")
    clinical_embeddings = [
        'complete_dataset_embeddings.npy',
        'complete_image_only_embeddings.npy', 
        'complete_text_only_embeddings.npy'
    ]
    
    for file in clinical_embeddings:
        total_tests += 1
        path = storage_path / 'embeddings' / file
        if path.exists():
            try:
                embeddings = np.load(path)
                if embeddings.shape[0] == 2006:
                    print(f"   ✅ {file}: shape {embeddings.shape}")
                    passed_tests += 1
                else:
                    print(f"   ❌ {file}: wrong size {embeddings.shape}")
            except:
                print(f"   ❌ {file}: corrupted")
        else:
            print(f"   ❌ {file}: missing")
    
    # Test 3: Literature embeddings
    print("3. Testing literature embeddings...")
    literature_files = [
        'literature_embeddings.npy',
        'literature_metadata.json',
        'literature_index.index',
        'literature_info.json'
    ]
    
    for file in literature_files:
        total_tests += 1
        path = storage_path / 'literature' / file
        if path.exists():
            try:
                if file.endswith('.npy'):
                    embeddings = np.load(path)
                    print(f"   ✅ {file}: shape {embeddings.shape}")
                elif file.endswith('.json'):
                    with open(path, 'r') as f:
                        data = json.load(f)
                    if file == 'literature_info.json' and 'total_papers' in data:
                        print(f"   ✅ {file}: {data['total_papers']} papers, {data['total_chunks']} chunks")
                    elif file == 'literature_metadata.json':
                        print(f"   ✅ {file}: {len(data)} chunks")
                    else:
                        print(f"   ✅ {file}: valid JSON")
                elif file.endswith('.index'):
                    index = faiss.read_index(str(path))
                    print(f"   ✅ {file}: {index.ntotal} vectors")
                passed_tests += 1
            except Exception as e:
                print(f"   ❌ {file}: error - {e}")
        else:
            print(f"   ❌ {file}: missing")
    
    # Test 4: Model files
    print("4. Testing model files...")
    model_files = ['cross_attention_best.pth', 'rag_query_encoder.pth']
    
    for file in model_files:
        total_tests += 1
        path = storage_path / 'models' / file
        if path.exists():
            try:
                checkpoint = torch.load(path, map_location='cpu')
                if 'model_state_dict' in checkpoint or 'rag_encoder_state_dict' in checkpoint:
                    print(f"   ✅ {file}: valid")
                    passed_tests += 1
                else:
                    print(f"   ❌ {file}: invalid format")
            except:
                print(f"   ❌ {file}: corrupted")
        else:
            print(f"   ❌ {file}: missing")
    
    # Test 5: FAISS indices
    print("5. Testing all FAISS indices...")
    index_files = [
        ('indices/complete_dataset_index.index', 2006),
        ('indices/image_only_index.index', 2006),
        ('indices/text_only_index.index', 2006),
        ('indices/candidate_index.index', None),
        ('literature/literature_index.index', None)
    ]
    
    for file_path, expected_count in index_files:
        total_tests += 1
        path = storage_path / file_path
        if path.exists():
            try:
                index = faiss.read_index(str(path))
                if expected_count and index.ntotal == expected_count:
                    print(f"   ✅ {file_path}: {index.ntotal} vectors")
                    passed_tests += 1
                elif not expected_count and index.ntotal > 0:
                    print(f"   ✅ {file_path}: {index.ntotal} vectors")
                    passed_tests += 1
                else:
                    print(f"   ❌ {file_path}: unexpected vector count {index.ntotal}")
            except:
                print(f"   ❌ {file_path}: corrupted")
        else:
            print(f"   ❌ {file_path}: missing")
    
    # Test 6: Dual retrieval test
    print("6. Testing dual retrieval (clinical + literature)...")
    total_tests += 2
    
    try:
        # Test clinical retrieval
        clinical_emb = np.load(storage_path / 'embeddings' / 'complete_dataset_embeddings.npy')
        clinical_idx = faiss.read_index(str(storage_path / 'indices' / 'complete_dataset_index.index'))
        query_vector = clinical_emb[0:1].astype('float32')
        scores, indices = clinical_idx.search(query_vector, k=3)
        
        if scores[0][0] > 0.9:
            print("   ✅ Clinical retrieval working")
            passed_tests += 1
        else:
            print("   ❌ Clinical retrieval failed")
            
        # Test literature retrieval
        lit_emb = np.load(storage_path / 'literature' / 'literature_embeddings.npy')
        lit_idx = faiss.read_index(str(storage_path / 'literature' / 'literature_index.index'))
        lit_query = lit_emb[0:1].astype('float32')
        scores, indices = lit_idx.search(lit_query, k=3)
        
        if scores[0][0] > 0.9:
            print("   ✅ Literature retrieval working")
            passed_tests += 1
        else:
            print("   ❌ Literature retrieval failed")
            
    except Exception as e:
        print(f"   ❌ Dual retrieval test failed: {e}")
    
    # Final report
    print("\n" + "=" * 50)
    print(f"STORAGE TEST RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ COMPLETE SYSTEM READY: Clinical Cases + Literature")
        return True
    else:
        print("❌ SYSTEM INCOMPLETE - CHECK MISSING COMPONENTS")
        return False

def quick_rag_simulation(storage_path="mammography_retrieval_storage"):
    """Simulate a quick RAG query to test end-to-end functionality"""
    
    print("\n8. RAG Simulation Test...")
    storage_path = Path(storage_path)
    
    try:
        # Load components
        embeddings = np.load(storage_path / 'embeddings' / 'complete_dataset_embeddings.npy')
        index = faiss.read_index(str(storage_path / 'indices' / 'complete_dataset_index.index'))
        
        with open(storage_path / 'metadata' / 'complete_dataset_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Simulate user query (use random sample as query)
        query_idx = 100
        query_embedding = embeddings[query_idx:query_idx+1].astype('float32')
        
        # Search
        scores, indices = index.search(query_embedding, k=3)
        
        print("   Query simulation:")
        print(f"   Query patient: {metadata[query_idx]['patient_id']}")
        print(f"   Query pathology: {metadata[query_idx]['pathology_label']}")
        print("   Top 3 results:")
        
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            result = metadata[idx]
            print(f"     {i+1}. Patient {result['patient_id']} ({result['pathology_label']}) - Score: {score:.4f}")
        
        print("   ✅ RAG simulation successful")
        return True
        
    except Exception as e:
        print(f"   ❌ RAG simulation failed: {e}")
        return False

if __name__ == "__main__":
    # Run comprehensive test
    storage_ok = test_complete_storage()
    
    if storage_ok:
        # Run RAG simulation
        rag_ok = quick_rag_simulation()
        
        if rag_ok:
            print("\n🎉 ALL TESTS PASSED - READY FOR RAG APP DEVELOPMENT")
        else:
            print("\n⚠️  STORAGE OK BUT RAG SIMULATION FAILED")
    else:
        print("\n❌ STORAGE INCOMPLETE - RETRAIN REQUIRED")