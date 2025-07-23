#!/usr/bin/env python3
"""
Test script for the multi-ligand API endpoint.
"""

import requests
import json
import time

# API Configuration
API_BASE_URL = "http://localhost:5000"

def test_multi_ligand_api():
    """Test the multi-ligand prediction API endpoint."""
    
    # Test data - same protein with multiple ligands
    test_data = {
        "protein_sequence": "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ",
        "ligands": [
            {
                "id": "0",
                "smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
            },
            {
                "id": "1", 
                "smiles": "CC(C)CC[C@H](N)C(=O)O"
            },
            {
                "id": "2",
                "smiles": "CC(C)(C)OC(=O)N[C@@H](CCCCN)C(=O)O"
            },
            # {
            #     "id": "3",
            #     "smiles": "N[C@@H](CC(=O)O)C(=O)O"
            # },
            # {
            #     "id": "4",
            #     "smiles": "CC[C@H](C)[C@@H](N)C(=O)O"
            # }
        ],
        "diffusion_samples": 1,
        "recycling_steps": 3,
        "sampling_steps": 200,
        "output_format": "mmcif"
    }
    
    print("=== Testing Multi-Ligand API ===")
    print(f"Testing with {len(test_data['ligands'])} ligands:")
    for ligand in test_data['ligands']:
        print(f"  - {ligand['id']}: {ligand['smiles']}")
    
    # Submit prediction
    print(f"\n🚀 Submitting multi-ligand prediction to {API_BASE_URL}/predict_multi")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict_multi",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        import pdb; pdb.set_trace()
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Multi-ligand prediction submitted successfully!")
            print(f"Job ID: {result.get('job_id')}")
            print(f"Status: {result.get('status')}")
            
            # Display summary
            summary = result.get('summary', {})
            print(f"\n📊 Summary:")
            print(f"  - Total ligands: {summary.get('total_ligands')}")
            print(f"  - Successful: {summary.get('successful')}")
            print(f"  - Failed: {summary.get('failed')}")
            print(f"  - Duration: {summary.get('duration', 0):.2f} seconds")
            
            # Display results for each ligand
            print(f"\n🧪 Individual Results:")
            results = result.get('results', {})
            for ligand_id, ligand_result in results.items():
                print(f"\n  {ligand_id}:")
                if 'error' in ligand_result:
                    print(f"    ❌ Error: {ligand_result['error']}")
                else:
                    print(f"    ✅ Success")
                    print(f"    Affinity: {ligand_result.get('affinity_pred_value', 'N/A')}")
                    print(f"    Binding Prob: {ligand_result.get('affinity_probability_binary', 'N/A')}")
                    print(f"    Duration: {ligand_result.get('duration', 0):.2f}s")
                    
                    files = ligand_result.get('files', {})
                    print(f"    Files: {len(files.get('structure', []))} structure, {len(files.get('affinity', []))} affinity")
            
            # Display comparative analysis
            analysis = result.get('analysis', {})
            if 'ranking' in analysis:
                print(f"\n🏆 Ligand Ranking (Best to Worst):")
                for rank_info in analysis['ranking']:
                    print(f"  {rank_info['rank']}. {rank_info['ligand_id']}: {rank_info['affinity_value']:.3f}")
                
                stats = analysis.get('statistics', {})
                if stats:
                    print(f"\n📈 Statistics:")
                    best = stats.get('best_affinity', {})
                    worst = stats.get('worst_affinity', {})
                    print(f"  - Best: {best.get('ligand_id')} ({best.get('value', 0):.3f})")
                    print(f"  - Worst: {worst.get('ligand_id')} ({worst.get('value', 0):.3f})")
                    print(f"  - Mean: {stats.get('mean_affinity', 0):.3f}")
                    print(f"  - Range: {stats.get('affinity_range', 0):.3f}")
            
            return result.get('job_id')
            
        else:
            print(f"❌ API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error making API request: {e}")
        return None

def test_single_ligand_api():
    """Test the single ligand API endpoint for comparison."""
    
    test_data = {
        "protein_sequence": "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ",
        "ligand_smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
    }
    
    print("\n=== Testing Single Ligand API (for comparison) ===")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Single ligand prediction successful!")
            print(f"Job ID: {result.get('job_id')}")
            
            results = result.get('results', {})
            print(f"Affinity: {results.get('affinity_pred_value', 'N/A')}")
            print(f"Binding Prob: {results.get('affinity_probability_binary', 'N/A')}")
            
            return result.get('job_id')
        else:
            print(f"❌ Single ligand API failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_job_retrieval(job_id):
    """Test retrieving job results."""
    if not job_id:
        return
    
    print(f"\n=== Testing Job Retrieval ===")
    print(f"Retrieving results for job: {job_id}")
    
    try:
        response = requests.get(f"{API_BASE_URL}/results/{job_id}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Job retrieval successful!")
            print(f"Status: {result.get('status')}")
            print(f"Type: {result.get('type', 'single_ligand')}")
            
            if result.get('type') == 'multi_ligand':
                summary = result.get('summary', {})
                print(f"Multi-ligand summary: {summary.get('successful')}/{summary.get('total_ligands')} successful")
            
        else:
            print(f"❌ Job retrieval failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error retrieving job: {e}")

def test_jobs_list():
    """Test listing all jobs."""
    print(f"\n=== Testing Jobs List ===")
    
    try:
        response = requests.get(f"{API_BASE_URL}/jobs")
        
        if response.status_code == 200:
            result = response.json()
            jobs = result.get('jobs', [])
            
            print(f"✅ Found {len(jobs)} jobs:")
            for job in jobs:
                print(f"  - {job.get('job_id')}: {job.get('type', 'single_ligand')} ({job.get('status')})")
                
        else:
            print(f"❌ Jobs list failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error listing jobs: {e}")

if __name__ == "__main__":
    print("🧪 Testing Boltz-2 Multi-Ligand API")
    print("=" * 50)
    
    # Test multi-ligand endpoint
    multi_job_id = test_multi_ligand_api()
    
    # Test single ligand endpoint for comparison
    single_job_id = test_single_ligand_api()
    
    # Test job retrieval
    test_job_retrieval(multi_job_id)
    
    # Test jobs listing
    test_jobs_list()
    
    print("\n✅ API testing complete!") 