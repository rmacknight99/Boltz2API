from Bio.PDB import PDBParser, PPBuilder
import requests, json, glob
from colorama import Fore, Style
import time, json, os
import uuid
import yaml
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, Tuple
import subprocess
import tqdm
import sys
import argparse

local_endpoints = {
    'boltz2': 'http://gpg-boltzmann.cheme.cmu.edu:5001/predict_multi'
}

def get_full_sequence(seq_dict):
    return ''.join([str(seq_dict[chain_id]) for chain_id in seq_dict.keys()])

def sequence_from_pdb(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    ppb = PPBuilder()
    
    sequence_dict = {}
    
    for model in structure:
        for chain in model:
            sequence = ""
            for pp in ppb.build_peptides(chain):
                sequence += pp.get_sequence()
            sequence_dict[chain.id] = sequence
    
    return get_full_sequence(sequence_dict)

def use_boltz2(protein_input, ligand_input, recycling_steps=3, sampling_steps=50, diffusion_samples=1, step_scale=1, top_k=10):
    print('\n--- Attempting to measure binding affinity of the ligand to the protein using Boltz2 ---\n', flush=True)

    payload = {}
    # Get the protein sequence
    if protein_input.endswith('.pdb'):
        protein_input = sequence_from_pdb(protein_input)
    
    payload['protein_sequence'] = protein_input
        
    ligand_inputs = [{"id": str(idx), "smiles": l.strip()} for idx, l in enumerate(ligand_input.split(','))]
    
    payload['ligands'] = ligand_inputs
    payload['diffusion_samples'] = diffusion_samples
    payload['recycling_steps'] = recycling_steps
    payload['sampling_steps'] = sampling_steps
    payload['step_scale'] = step_scale
    payload['output_format'] = 'pdb'
    
    print(json.dumps(payload, indent=4), flush=True)
    
    output_str = ""
    
    print(f"--- Sending {len(ligand_inputs)} ligands to Boltz2 ---\n", flush=True)
    try:
        response = requests.post(local_endpoints['boltz2'], json=payload, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            result = response.json()
        else:
            print(Fore.RED + f'An error occurred while running Boltz2.' + Style.RESET_ALL, flush=True)
            result = None
            # Fixed error handling
            try:
                error_msg = response.json().get('error', 'Unknown error')
            except:
                error_msg = f"HTTP {response.status_code}"
            output_str = f"An error occurred while running Boltz2. Status code: {response.status_code}. Error: {error_msg}"
    except Exception as e:
        result = None
        output_str = f"An error occurred while running Boltz2. Error: {e}"
    
    # Process the results
    scores = {}
    ligand_results = []
    if result is not None:
        output_str += f"Boltz2 ran successfully!\nHere are the results:\n"
        results = result.get('results', result.get('result'))
        for ligand_input in ligand_inputs:
            ligand_id = ligand_input['id']
            ligand_smiles = ligand_input['smiles']
            ligand_result = results[ligand_id]

            ligand_results.append({'id': ligand_id, 'smiles': ligand_smiles, 'result': ligand_result})
            
            # Process the affinity data
            aff_data = ligand_result['affinity_data']
            if aff_data is None:
                continue

            pred_aff = aff_data['affinity_pred_value']
            ic50_uM = 10**(pred_aff) * 10**3
            # Add some better information for the ligand result
            ligand_result['ic50_uM'] = ic50_uM
            ligand_result['ligand_smiles'] = ligand_smiles
            ligand_result['binding_prob'] = aff_data['affinity_probability_binary']
            
            # Process the confidence data
            conf_data = ligand_result['confidence_data']
            confidence_score = conf_data['confidence_score']
            ligand_result['confidence_score'] = confidence_score
            
            results[ligand_id] = ligand_result
            # Score should prioritize low IC50 (strong binding), high confidence, and high binding probability
            score = (confidence_score * ligand_result['binding_prob']) / ic50_uM
            scores[ligand_id] = score
            
        # Sort the scores by score (higher score = better)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (ligand_id, score) in enumerate(sorted_scores[:top_k]):
            ligand_result = results[ligand_id]
            ligand_smiles = ligand_result['ligand_smiles']  # Use the stored smiles for this specific ligand
            output_str += f"Rank {rank+1}: {ligand_smiles}\n"
            output_str += f"   Binding Probability (0-1): {ligand_result['binding_prob']:.2f}\n"
            output_str += f"   Confidence Score (0-1): {ligand_result['confidence_score']:.2f}\n"
            output_str += f"   Predicted IC50 (uM): {ligand_result['ic50_uM']:.2f}\n"
            output_str += f"   **NOTE** The confidence score is a measure of how confident the model is in the protein-ligand complex used for the predictions.\n"
            output_str += f"   **NOTE** Low IC50 --> strong binding, high IC50 --> weak binding.\n"
            # output_str += f"   **NOTE** If the binding probability is low, the predicted IC50 and pIC50 are likely to be unreliable.\n"
            output_str += "-"*75 + "\n"
    
    return output_str, ligand_results

def generate_job_id() -> str:
    """Generate a short unique job ID."""
    return str(uuid.uuid4())[:8]

def get_amide_filename(smiles: str) -> str:
    """Convert amide SMILES to a safe filename."""
    # Mapping for known amides
    amide_names = {
        "O=C(NCCCCCCCC)C1=CC=CC=C1": "N_Octylbenzamide",
        "O=C(NC1=CC=CC=C1F)C2=CC=CC=C2": "N_2_Fluorophenylbenzamide", 
        "O=C(NC1=CC=C(CCCCCCC)C=C1)C2=CC=CC=C2": "N_4_Heptylphenylbenzamide",
        "O=C(NCCCN)C1=CC=CC=C1": "N_3_Aminopropylbenzamide"
    }
    
    if smiles in amide_names:
        return amide_names[smiles]
    else:
        # Create a safe filename from SMILES
        safe_name = smiles.replace("=", "_").replace("(", "_").replace(")", "_")
        safe_name = safe_name.replace("[", "_").replace("]", "_").replace("#", "_")
        safe_name = safe_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        return f"amide_{safe_name[:20]}"  # Truncate to avoid long filenames

def run_single_ligand_prediction(config_file: Path, job_id: str, ligand_id: str, options: Dict[str, Any], device_id: int = 0) -> Tuple[bool, str, Dict[str, Any]]:
    """Run Boltz-2 prediction for a single ligand in parallel processing."""
    
    output_dir = RESULTS_DIR / job_id / ligand_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # print(f'Running Boltz-2 prediction for ligand {ligand_id} on GPU {device_id}, results will be saved in {output_dir}')
    
    # Set up environment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(device_id)
    # env["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64/stubs:{env.get('LD_LIBRARY_PATH', '')}"
    
    # Run boltz command
    cmd = [
        "boltz", "predict", str(config_file),
        "--use_msa_server",
        "--out_dir", str(output_dir),
        "--no_kernels",
        "--devices", "1",
        "--accelerator", "gpu",
        "--output_format", options.get("output_format", "mmcif"),
        "--diffusion_samples", str(options.get("diffusion_samples", 1)),
        "--diffusion_samples_affinity", "5",
        "--sampling_steps", str(options.get("sampling_steps", 200)),
        "--recycling_steps", str(options.get("recycling_steps", 3)),
        "--max_parallel_samples", "1",
        "--affinity_mw_correction",
        "--write_full_pae",
        "--num_workers", "2",
    ]
    
    start_time = time.time()
    
    # print(f'RUNNING BOLTZ COMMAND:\n{' '.join(cmd)}')
    
    try:
        output = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout per ligand
        )
        
        end_time = time.time()
        duration = end_time - start_time
        if output.returncode == 0:
            result = parse_results(output_dir / f"boltz_results_{job_id}_{ligand_id}", ligand_id, job_id)
            result["duration"] = duration
            return True, ligand_id, result
        else:
            return False, ligand_id, {"error": output.stderr, "duration": duration}
            
    except subprocess.TimeoutExpired:
        return False, ligand_id, {"error": "timeout", "duration": time.time() - start_time}
    except Exception as e:
        return False, ligand_id, {"error": str(e), "duration": time.time() - start_time}

def parse_results(results_dir: Path, ligand_id: str, job_id: str) -> Dict[str, Any]:
    """Parse Boltz-2 results from the output directory."""
    predictions_dir = results_dir / "predictions" / f"{job_id}_{ligand_id}"
    
    result = {}
    
    if not predictions_dir.exists():
        return {"error": "No predictions directory found"}
    
    try:
        affinity_file = str(list(predictions_dir.glob("affinity*.json"))[0])
        with open(affinity_file, 'r') as f:
            affinity_data = json.load(f)
        result['affinity_data'] = affinity_data
    except Exception as e:
        print(f'Error parsing affinity data: {e}', flush=True)
        result['affinity_data'] = None
    
    try:
        structure_files = list(predictions_dir.glob("*.cif")) + list(predictions_dir.glob("*.pdb"))
        structure_file = str(structure_files[0])
        structure_data = open(structure_file, 'r').read()
        result['structure_data'] = structure_data
    except Exception as e:
        print(f'Error parsing structure data: {e}', flush=True)
        result['structure_data'] = None
    
    try:
        confidence_file = str(list(predictions_dir.glob("confidence*.json"))[0])
        with open(confidence_file, 'r') as f:
            confidence_data = json.load(f)
        result['confidence_data'] = confidence_data
    except Exception as e:
        print(f'Error parsing confidence data: {e}', flush=True)
        result['confidence_data'] = None
    
    result['data_keys'] = list(result.keys())
    return result

def use_boltz2_local(data):
    job_id = generate_job_id()
    protein_sequence = data.get("protein_sequence")
    ligands = data.get("ligands")

    options = {
        "diffusion_samples": data.get("diffusion_samples", 1),
        "recycling_steps": data.get("recycling_steps", 3),
        "sampling_steps": data.get("sampling_steps", 200),
        "output_format": data.get("output_format", "mmcif")
    }

    config_files = []
    ligand_ids = []

    for i, ligand in enumerate(ligands):
        if isinstance(ligand, dict):
            ligand_id = ligand.get("id", f"L{i+1}")
            ligand_smiles = ligand.get("smiles")
        else:
            ligand_id = f"L{i+1}"
            ligand_smiles = ligand
        
        if not ligand_smiles:
            return {"error": f"Invalid ligand format at index {i}"}
        
        config_file = create_config_file(protein_sequence, ligand_smiles, ligand_id, job_id)
        config_files.append(config_file)
        ligand_ids.append(ligand_id)

    start_time = time.time()

    gpu_assignments = {}
    for i, ligand_id in enumerate(ligand_ids):
        gpu_id = AVAILABLE_GPUS[i % NUM_GPUS]
        gpu_assignments[ligand_id] = gpu_id
    
    # print(f'Processing {len(ligand_ids)} ligands on {NUM_GPUS} GPUs')
    # print(f'GPU assignments: {gpu_assignments}')

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
            # Submit all jobs with their assigned GPU
            future_to_ligand = {
                executor.submit(run_single_ligand_prediction, config_file, job_id, ligand_id, options, gpu_assignments[ligand_id]): ligand_id
                for config_file, ligand_id in zip(config_files, ligand_ids)
            }
            
            results = {}
            successful_ligands = []
            failed_ligands = []
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_ligand):
                ligand_id = future_to_ligand[future]
                try:
                    success, returned_ligand_id, result_data = future.result()
                    if success:
                        results[ligand_id] = result_data
                        successful_ligands.append(ligand_id)
                    else:
                        # print(f'Error running Boltz-2 for ligand {ligand_id}: {result_data.get("error", "Unknown error")}')
                        results[ligand_id] = {"error": result_data.get("error", "Unknown error")}
                        failed_ligands.append(ligand_id)
                except Exception as e:
                    results[ligand_id] = {"error": str(e)}
                    failed_ligands.append(ligand_id)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        results['total_duration'] = total_duration
        results['job_id'] = job_id
        results['protein_sequence'] = protein_sequence
        results['ligands'] = ligands
        results['options'] = options
        results['status'] = "completed" if successful_ligands else "failed"
        results['timestamp'] = time.time()
        results['data_keys'] = list(results.keys())
        
        return {
            "job_id": job_id,
            "status": "completed",
            "result": results,
        }
        
    finally:
        # Clean up temp config files
        for config_file in config_files:
            config_file.unlink(missing_ok=True)
    
def create_config_file(protein_sequence: str, ligand_smiles: str, ligand_id: str, job_id: str) -> Path:
    """Create a Boltz-2 configuration file for a single ligand."""
    config = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": protein_sequence
                }
            },
            {
                "ligand": {
                    "id": ligand_id,
                    "smiles": ligand_smiles
                }
            }
        ],
        "properties": [
            {
                "affinity": {
                    "binder": ligand_id
                }
            }
        ]
    }
    
    config_file = TEMP_DIR / f"{job_id}_{ligand_id}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return config_file

def test_boltz2_ranking():
    """Test use_boltz2 with multiple ligands to validate ranking"""
    
    # Test PDB file (protein target)
    test_pdb = "/home/rmacknig/boltz2_local/4WUN.pdb"
    
    # Multiple test ligands (SMILES strings)
    # Mix of different types to see ranking diversity
    ligand_inputs = [
        "CC1=C(C=C(C=C1)C(=O)NC2=CC(=C(C=C2)CN3CCN(CC3)C)C(F)(F)F)C#CC4=CN=C5N4N=CC=C5",  # Ponatinib
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",  # Imantinib
        "C1=CC=C(C=C1)C(=O)C2=CC=CC=C2",  # Benzophenone derivative
        "O=C(O)C1=CC=CC=C1",  # Benzoic acid - simple
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CC1=CC=C(C=C1)C(=O)C2=CC=CC=C2",  # Morphine
        "CN1C=NC2=C1C(=O)NC(C(C2=O)C3=CC=CC=C3)=O",
        "CCO",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(=O)NC1=CC=C(C=C1)C(=O)O", # acetamide
        "C1=CC=C(C=C1)C(=O)C2=CC=CC=C2", # acetophenone
    ]
    
    # Convert to comma-separated string as expected by use_boltz2
    ligand_input_str = ",".join(ligand_inputs)
    
    print("="*80)
    print("TESTING use_boltz2 RANKING")
    print("="*80)
    print(f"Protein: {os.path.basename(test_pdb)}")
    print(f"Number of ligands: {len(ligand_inputs)}")
    print(f"Ligands:")
    for i, ligand in enumerate(ligand_inputs, 1):
        print(f"  {i}. {ligand}")
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    
    try:
        result = use_boltz2(test_pdb, ligand_input_str, top_k=len(ligand_inputs))
        print(result)
        
        print("\n" + "="*80)
        print("TEST EVALUATION:")
        print("="*80)
        print("✅ SUCCESS: use_boltz2 completed without errors")
        print("✅ Check that rankings make sense:")
        print("   - Lower IC50 values should rank higher")
        print("   - Higher confidence scores should boost ranking")
        print("   - Higher binding probabilities should boost ranking")
        print("   - The scoring formula: (confidence × binding_prob) / IC50")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("\nThis might be due to:")
        print("- Boltz2 API not being available")
        print("- Missing environment variables")
        print("- Network connectivity issues")
        
    print("\n" + "="*80)

def test_boltz2_single_ligand():
    """Test with single ligand (simpler test)"""
    test_pdb = "/home/rmacknig/scientificOS/runs/amide_couplings_claude-sonnet-4-20250514_trial_1_targets/data/pdb_structures/raw/4WUN.pdb"
    ligand_input = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin - simple test case
    
    print("="*80)
    print("TESTING use_boltz2 WITH SINGLE LIGAND")
    print("="*80)
    print(f"Protein: {os.path.basename(test_pdb)}")
    print(f"Ligand: {ligand_input}")
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    
    try:
        result = use_boltz2(test_pdb, ligand_input)
        print(result)
        print("✅ SUCCESS: Single ligand test completed")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("\n" + "="*80)

def cleanup_directories(results_dir, temp_dir):
    """Clean up results and temp directories to save disk space."""
    try:
        import shutil
        # Only clean up if the directories belong to this specific job
        if results_dir.exists() and str(results_dir).endswith(UNIQUE_JOB_ID):
            shutil.rmtree(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            print(f"🧹 Cleaned up job-specific results: {results_dir}", flush=True)
        
        if temp_dir.exists() and str(temp_dir).endswith(UNIQUE_JOB_ID):
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            print(f"🧹 Cleaned up job-specific temp: {temp_dir}", flush=True)
    except Exception as e:
        print(f"⚠️  Warning: Could not clean directories: {e}", flush=True)

def final_cleanup(results_dir, temp_dir):
    """Final cleanup at the end of the job."""
    try:
        import shutil
        print(f"🧹 Final cleanup for job {UNIQUE_JOB_ID}...", flush=True)
        
        if results_dir.exists():
            shutil.rmtree(results_dir)
            print(f"🗑️  Removed job-specific results directory: {results_dir}", flush=True)
        
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"🗑️  Removed job-specific temp directory: {temp_dir}", flush=True)
            
    except Exception as e:
        print(f"⚠️  Warning: Could not perform final cleanup: {e}", flush=True)

def create_continue_flag(amides, pdbs, already_done_tuples):
    """Create continue flag based on remaining work."""
    # These variables need to be accessible here
    total_work = len(amides) * len(pdbs)
    completed_work = len(already_done_tuples)
    remaining_work = total_work - completed_work
    
    if remaining_work > 0:
        with open("continue_flag.txt", "w") as f:
            f.write(f"Remaining: {remaining_work}/{total_work} combinations\n")
            f.write(f"Progress: {completed_work/total_work*100:.1f}%\n")
        return True
    else:
        if os.path.exists("continue_flag.txt"):
            os.remove("continue_flag.txt")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run Boltz2 predictions for amide-protein binding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python boltz_predict.py 2                                       # Use 2 GPUs for all amides
  python boltz_predict.py 2 --dry-run                             # Show what would be processed
  python boltz_predict.py 1 --amide 'O=C(NCCCCCCCC)C1=CC=CC=C1'   # Run specific amide only
  python boltz_predict.py 2 --chembl-file target_chembl_ids.txt   # Filter by ChEMBL IDs from file
        """
    )
    
    parser.add_argument('num_gpus', type=int, 
                       help='Number of GPUs to use for parallel processing')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show diagnostics without running actual predictions')  
    parser.add_argument('--amide', type=str,
                       help='Run predictions for specific amide SMILES only')
    parser.add_argument('--chembl-file', type=str,
                       help='File containing ChEMBL IDs to filter (one per line)')
    
    args = parser.parse_args()
    
    # Create unique directories for this job to avoid conflicts with parallel jobs
    import time
    JOB_TIMESTAMP = str(int(time.time()))
    JOB_PID = str(os.getpid())
    UNIQUE_JOB_ID = f"{JOB_TIMESTAMP}_{JOB_PID}"
    
    RESULTS_DIR = Path(f"/project/flame/rmacknig/boltz2_configs/orchard_boltz2_results_{UNIQUE_JOB_ID}")
    TEMP_DIR = Path(f"/project/flame/rmacknig/boltz2_configs/orchard_boltz2_temp_{UNIQUE_JOB_ID}")

    # Make the directories if they don't exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f'📁 Job-specific directories created:', flush=True)
    print(f'   Results: {RESULTS_DIR}', flush=True)
    print(f'   Temp: {TEMP_DIR}', flush=True)

    NUM_GPUS = args.num_gpus
    DRY_RUN = args.dry_run
    SPECIFIC_AMIDE = args.amide
    
    # Load ChEMBL IDs from file if specified
    TARGET_CHEMBL_IDS = None
    if args.chembl_file:
        if os.path.exists(args.chembl_file):
            with open(args.chembl_file, 'r') as f:
                TARGET_CHEMBL_IDS = set(line.strip() for line in f if line.strip() and not line.strip().startswith('#'))
            print(f'📋 Loaded {len(TARGET_CHEMBL_IDS)} target ChEMBL IDs from {args.chembl_file}', flush=True)
            print(f'   ChEMBL IDs: {sorted(list(TARGET_CHEMBL_IDS))[:10]}{"..." if len(TARGET_CHEMBL_IDS) > 10 else ""}', flush=True)
        else:
            print(f'❌ Error: ChEMBL file not found: {args.chembl_file}', flush=True)
            sys.exit(1)
    
    AVAILABLE_GPUS = list(range(NUM_GPUS))
    
    if DRY_RUN:
        print('🔍 DRY RUN MODE - Will show diagnostics but not run actual predictions', flush=True)
    else:
        print('🚀 LIVE MODE - Will run actual Boltz2 predictions', flush=True)

    # Load amides
    all_amides = [a.strip().split(',')[0] for a in open("./amides.txt").readlines()]
    all_amide_names = [a.strip().split(',')[1] for a in open("./amides.txt").readlines()]
    
    if SPECIFIC_AMIDE:
        if SPECIFIC_AMIDE in all_amides:
            amides = [SPECIFIC_AMIDE]
            amide_filename = all_amide_names[all_amides.index(SPECIFIC_AMIDE)]
            print(f'🎯 SPECIFIC AMIDE MODE: Running predictions for {amide_filename}', flush=True)
            print(f'   SMILES: {SPECIFIC_AMIDE}', flush=True)
        else:
            print(f'❌ ERROR: Specified amide not found in amides.txt', flush=True)
            print(f'   Requested: {SPECIFIC_AMIDE}', flush=True)
            print(f'   Available amides:', flush=True)
            for amide in all_amides:
                print(f'     {amide}', flush=True)
            sys.exit(1)
    else:
        amides = all_amides
        
        amide_filename = "all_amides"
        print(f'📋 ALL AMIDES MODE: Loaded {len(amides)} amides', flush=True)
    
    # Load PDB length data for filtering
    pdb_to_length = {}
    if os.path.exists("./pdb_to_length.json"):
        with open("./pdb_to_length.json", 'r') as f:
            pdb_to_length = json.load(f)
        print(f'Loaded length data for {len(pdb_to_length)} PDBs', flush=True)
    else:
        print('⚠️  Warning: pdb_to_length.json not found, proceeding without length filtering', flush=True)
    
    # Load all PDBs and filter by ChEMBL ID if specified
    all_pdbs = glob.glob("../pdb_files_by_chembl_id/*/*_clean.pdb")
    print(f'Found {len(all_pdbs)} total PDBs', flush=True)
    
    # Filter by ChEMBL ID if file was provided
    if TARGET_CHEMBL_IDS:
        filtered_pdbs = []
        chembl_filtered_count = 0
        for pdb_file in all_pdbs:
            # Extract ChEMBL ID from path: ../pdb_files_by_chembl_id/CHEMBLXXX/...
            chembl_id = pdb_file.split('/')[-2]  # Get directory name
            if chembl_id in TARGET_CHEMBL_IDS:
                filtered_pdbs.append(pdb_file)
            else:
                chembl_filtered_count += 1
        
        print(f'After ChEMBL ID filtering: {len(filtered_pdbs)}/{len(all_pdbs)} PDBs remaining', flush=True)
        print(f'  🚫 PDBs filtered out by ChEMBL ID: {chembl_filtered_count}', flush=True)
        all_pdbs = filtered_pdbs
    
    # Filter PDBs to only include those with length < 1000
    pdbs = []
    filtered_out_count = 0
    no_length_data_count = 0
    
    for pdb_file in all_pdbs:
        if pdb_file in pdb_to_length:
            length = pdb_to_length[pdb_file]
            if length < 1000:
                pdbs.append(pdb_file)
            else:
                filtered_out_count += 1
        else:
            no_length_data_count += 1
            # If no length data available, include the PDB (conservative approach)
            pdbs.append(pdb_file)
    
    print(f'After length filtering (< 1000 residues):', flush=True)
    print(f'  ✅ PDBs to process: {len(pdbs)}', flush=True)
    print(f'  🚫 PDBs filtered out (≥ 1000 residues): {filtered_out_count}', flush=True)
    if no_length_data_count > 0:
        print(f'  ❓ PDBs with no length data (included): {no_length_data_count}', flush=True)
    
    unique_pdbs = set(pdbs)
    print(f'Unique PDBs after filtering: {len(unique_pdbs)}', flush=True)

    # Set up output filename - save to /project/flame/rmacknig/ to avoid home directory disk quota
    # Include ChEMBL filtering info in filename if applicable
    filename_suffix = amide_filename
    if TARGET_CHEMBL_IDS:
        chembl_count = len(TARGET_CHEMBL_IDS)
        filename_suffix += f"_chembl_{chembl_count}ids"
    
    predictions_file = f"/project/flame/rmacknig/predictions_{filename_suffix}.json"
    print(f'📁 Results will be saved to: {predictions_file}', flush=True)
    
    log_data = []

    # Always check base predictions.json first to avoid duplicating work
    base_predictions_file = "/project/flame/rmacknig/predictions.json"
    if os.path.exists(base_predictions_file):
        base_predictions = json.load(open(base_predictions_file))
        print(f'📂 Loaded {len(base_predictions)} existing predictions from base file: {base_predictions_file}', flush=True)
        
        if SPECIFIC_AMIDE:
            # Filter base predictions for only the specific amide
            relevant_predictions = []
            for pred in base_predictions:
                amide_list = pred['amide']
                if isinstance(amide_list, list):
                    if SPECIFIC_AMIDE in amide_list:
                        relevant_predictions.append(pred)
                else:
                    if amide_list == SPECIFIC_AMIDE:
                        relevant_predictions.append(pred)
            
            log_data = relevant_predictions
            print(f'🎯 Filtered to {len(log_data)} predictions relevant to specific amide', flush=True)
        else:
            log_data = base_predictions
    
    # Also load amide-specific file if it exists (for additional data)
    if os.path.exists(predictions_file) and predictions_file != base_predictions_file:
        amide_specific_predictions = json.load(open(predictions_file))
        print(f'📂 Found {len(amide_specific_predictions)} predictions in amide-specific file: {predictions_file}', flush=True)
        
        # Merge with base predictions, avoiding duplicates
        existing_keys = set()
        for pred in log_data:
            key = (str(pred['amide']), pred['pdb'])
            existing_keys.add(key)
        
        added_count = 0
        for pred in amide_specific_predictions:
            key = (str(pred['amide']), pred['pdb'])
            if key not in existing_keys:
                log_data.append(pred)
                existing_keys.add(key)
                added_count += 1
        
        if added_count > 0:
            print(f'➕ Added {added_count} unique predictions from amide-specific file', flush=True)
    
    if not log_data:
        print(f'📝 Starting fresh - no existing predictions found', flush=True)

    already_done_tuples = []
    unique_pdbs = set()
    error_entries_skipped = 0
    
    for log in log_data:
        amides_ = log['amide']
        unique_pdbs.add(log['pdb'])
        
        # Check if this entry has any errors and should be skipped
        has_errors = False
        if 'data' in log and 'result' in log['data']:
            result = log['data']['result']
            for ligand_id in ['L1', 'L2']:
                if ligand_id in result and isinstance(result[ligand_id], dict):
                    if 'error' in result[ligand_id]:
                        has_errors = True
                        break
        
        if has_errors:
            error_entries_skipped += 1
            continue  # Skip this entry - don't count as "already done"
        
        # Only count as "already done" if no errors
        if not isinstance(amides_, list):
            already_done_tuples.append((amides_, log['pdb']))
        else:
            for amide in amides_:
                already_done_tuples.append((amide, log['pdb']))
    
    if error_entries_skipped > 0:
        print(f'⚠️  Skipped {error_entries_skipped} entries with errors (will be retried)', flush=True)
    print(f'\n{"="*80}', flush=True)
    print('DATA LOADING & DEDUPLICATION ANALYSIS', flush=True)
    print(f'{"="*80}', flush=True)
    print(f'📂 Loaded {len(log_data)} prediction records from predictions.json', flush=True)
    print(f'🧬 Total PDBs available: {len(pdbs)}', flush=True)
    print(f'🧪 Total amides to process: {len(amides)}', flush=True)
    print(f'🔄 Raw (amide, PDB) tuples from file: {len(already_done_tuples)}', flush=True)
    print(f'📊 Unique PDBs in predictions: {len(unique_pdbs)}', flush=True)
    
    unique_tuples = set(already_done_tuples)
    print(f'✨ Unique (amide, PDB) combinations: {len(unique_tuples)}', flush=True)
    
    if len(already_done_tuples) != len(unique_tuples):
        duplicates = len(already_done_tuples) - len(unique_tuples)
        print(f'⚠️  WARNING: Found {duplicates} duplicate entries ({duplicates/len(already_done_tuples)*100:.1f}% duplication rate)!', flush=True)
        print(f'🔧 Using deduplicated tuples for calculations...', flush=True)
        already_done_tuples = list(unique_tuples)
    else:
        print(f'✅ No duplicates found - data is clean!', flush=True)
    
    print(f'{"="*80}', flush=True)

    # batch amides by NUM_GPUS
    if NUM_GPUS == 1:
        batched_amides = [[a] for a in amides]
    else:
        batched_amides = [amides[i::NUM_GPUS] for i in range(NUM_GPUS)]
    pdbs_to_run_for_batch = {"\n".join(amides): [] for amides in batched_amides}

    for amides in batched_amides:
        total_skipped = 0
        for pdb_file in pdbs:
            already_done = [(amide, pdb_file) in already_done_tuples for amide in amides]
            if all(already_done):
                total_skipped += 1
                continue
            else:
                pdbs_to_run_for_batch["\n".join(amides)].append(pdb_file)
        print(f'Batch {amides}: skipped {total_skipped} PDBs where ALL amides in batch were already done', flush=True)
    
    # Detailed diagnostics for each amide
    print(f'\n{"="*80}', flush=True)
    print('DETAILED DIAGNOSTICS BY AMIDE', flush=True)
    print(f'{"="*80}', flush=True)
    
    total_needed = 0
    total_done = 0
    
    for amides in batched_amides:
        for amide in amides:
            already_done = 0
            for t in already_done_tuples:
                if amide in t:
                    already_done += 1
            
            needed = len(pdbs) - already_done
            total_needed += needed
            total_done += already_done
            
            print(f'Amide: {amide}', flush=True)
            print(f'  ✅ Already done: {already_done}/{len(pdbs)} PDBs ({already_done/len(pdbs)*100:.1f}%)', flush=True)
            print(f'  🔄 Still needed: {needed} PDBs', flush=True)
            print(f'  📊 Progress: {"█" * int(already_done/len(pdbs)*20)}{"░" * (20-int(already_done/len(pdbs)*20))} {already_done/len(pdbs)*100:.1f}%', flush=True)
            print(flush=True)
    
    print(f'OVERALL PROGRESS SUMMARY:', flush=True)
    print(f'  Total combinations needed: {len(amides) * len(pdbs)}', flush=True)
    print(f'  Total combinations done: {total_done}', flush=True)
    print(f'  Total combinations remaining: {total_needed}', flush=True)
    print(f'  Overall progress: {total_done/(len(amides) * len(pdbs))*100:.1f}%', flush=True)
    print(f'{"="*80}\n', flush=True)
    
    # Batch-level diagnostics
    print(f'BATCH PROCESSING PLAN:', flush=True)
    print(f'{"="*80}', flush=True)
    total_batch_work = 0
    
    for batch_idx, amides in enumerate(batched_amides):
        pdbs_to_run = pdbs_to_run_for_batch["\n".join(amides)]
        combinations_to_process = len(pdbs_to_run) * len(amides)
        total_batch_work += combinations_to_process
        
        print(f'Batch {batch_idx + 1}/{len(batched_amides)} (GPU{batch_idx % NUM_GPUS}):', flush=True)
        print(f'  Amides: {amides}', flush=True)
        print(f'  PDBs to process: {len(pdbs_to_run)}/{len(pdbs)} ({len(pdbs_to_run)/len(pdbs)*100:.1f}%)', flush=True)
        print(f'  Combinations to process: {combinations_to_process}', flush=True)
        
        if len(pdbs_to_run) == 0:
            print(f'  🎉 STATUS: Complete - no work needed!', flush=True)
        else:
            est_time_min = combinations_to_process * 2  # rough estimate: 2 min per combination
            print(f'  ⏱️  Estimated time: ~{est_time_min} minutes ({est_time_min/60:.1f} hours)', flush=True)
        print(flush=True)
    
    print(f'TOTAL WORK REMAINING:', flush=True)
    print(f'  Total combinations to process: {total_batch_work}', flush=True)
    if total_batch_work > 0:
        total_est_time_min = total_batch_work * 2
        print(f'  Estimated total time: ~{total_est_time_min} minutes ({total_est_time_min/60:.1f} hours)', flush=True)
    print(f'{"="*80}\n', flush=True)

    if DRY_RUN:
        print(f'\n🔍 DRY RUN COMPLETE - Here is what would be processed:', flush=True)
        print(f'{"="*80}', flush=True)
        total_work = sum(len(pdbs_to_run_for_batch["\n".join(amides)]) * len(amides) for amides in batched_amides)
        print(f'📋 WORK SUMMARY:', flush=True)
        for batch_idx, amides in enumerate(batched_amides):
            pdbs_to_run = pdbs_to_run_for_batch["\n".join(amides)]
            combinations = len(pdbs_to_run) * len(amides)
            print(f'  Batch {batch_idx + 1}: {len(pdbs_to_run)} PDBs × {len(amides)} amides = {combinations} combinations', flush=True)
        print(f'  TOTAL: {total_work} combinations to process', flush=True)
        print(f'\n🚀 To run actual predictions, use: python boltz_predict.py {NUM_GPUS}', flush=True)
        print(f'{"="*80}', flush=True)
    else:
        # Process each batch
        for batch_idx, amides in enumerate(batched_amides):
            pdbs_to_run = pdbs_to_run_for_batch["\n".join(amides)]
            print(f'🚀 STARTING batch {batch_idx + 1}/{len(batched_amides)} (GPU{batch_idx % NUM_GPUS}): {len(pdbs_to_run)} PDBs for amides: `{amides}`', flush=True)
            
            if not pdbs_to_run:
                print(f'No PDBs to process for batch {batch_idx + 1}, skipping...', flush=True)
                continue
                
            processed_count = 0
            
            for pdb_idx, pdb_file in enumerate(tqdm.tqdm(pdbs_to_run, desc=f"Batch {batch_idx + 1}")):
                try:
                    print(f'Processing PDB {pdb_idx + 1}/{len(pdbs_to_run)}: {os.path.basename(pdb_file)}', flush=True)
                    
                    protein_seq = sequence_from_pdb(pdb_file)
                    data = {
                        "protein_sequence": protein_seq,
                        "ligands": amides,
                        "diffusion_samples": 1,
                        "recycling_steps": 3,
                        "sampling_steps": 200,
                        "output_format": "mmcif"
                    }

                    output = use_boltz2_local(data)

                    log_entry = {
                        "amide": amides,
                        "pdb": pdb_file,
                        "data": output
                    }

                    log_data.append(log_entry)
                    processed_count += 1
                    
                    # Save progress every 5 PDBs
                    if processed_count % 5 == 0:
                        with open(predictions_file, "w") as f:
                            json.dump(log_data, f, indent=4)
                        print(f'💾 Saved progress after {processed_count} PDBs to {predictions_file}', flush=True)
                    
                    # Clean up directories every 20 PDBs to avoid disk quota
                    if processed_count % 20 == 0:
                        print(f'🧹 Cleaning up after {processed_count} PDBs...', flush=True)
                        cleanup_directories(RESULTS_DIR, TEMP_DIR)
                        
                except Exception as e:
                    print(f'❌ Error processing {pdb_file}: {e}', flush=True)
                    # Continue with next PDB instead of crashing
                    continue
            
            # Final save for this batch
            with open(predictions_file, "w") as f:
                json.dump(log_data, f, indent=4)
            print(f'✅ Completed batch {batch_idx + 1}: processed {processed_count} PDBs', flush=True)
        
        # Final summary for live mode
        print(f'\n{"="*80}', flush=True)
        print('🎉 ALL BATCHES COMPLETED!', flush=True)
        print(f'{"="*80}', flush=True)
        
        # Re-count final stats
        final_log_data = json.load(open(predictions_file)) if os.path.exists(predictions_file) else []
        final_tuples = []
        for log in final_log_data:
            amides_ = log['amide']
            if not isinstance(amides_, list):
                final_tuples.append((amides_, log['pdb']))
            else:
                for amide in amides_:
                    final_tuples.append((amide, log['pdb']))
        
        unique_final_tuples = set(final_tuples)
        total_possible = len(amides) * len(pdbs)
        
        print(f'📊 FINAL STATISTICS:', flush=True)
        print(f'  Total possible combinations: {total_possible}', flush=True)
        print(f'  Combinations completed: {len(unique_final_tuples)}', flush=True)
        print(f'  Completion rate: {len(unique_final_tuples)/total_possible*100:.1f}%', flush=True)
        print(f'  Results saved to: {predictions_file}', flush=True)
        print(f'{"="*80}', flush=True)
        
        # Final cleanup of job-specific directories
        final_cleanup(RESULTS_DIR, TEMP_DIR)