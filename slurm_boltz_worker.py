#!/usr/bin/env python3
"""
SLURM worker script for Boltz-2 orchard predictions.
This script runs the actual Boltz-2 predictions on the cluster nodes.
"""

import os
import sys
import json
import time
import glob
import shutil
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Tuple
import subprocess
import yaml
from Bio.PDB import PDBParser, PPBuilder

def get_full_sequence(seq_dict):
    """Get full sequence from sequence dictionary."""
    return ''.join([str(seq_dict[chain_id]) for chain_id in seq_dict.keys()])

def sequence_from_pdb(pdb_path: str):
    """Extract protein sequence from PDB file."""
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

def create_config_file(protein_sequence: str, ligand_smiles: str, ligand_id: str, job_id: str, temp_dir: Path) -> Path:
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
    
    config_file = temp_dir / f"{job_id}_{ligand_id}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return config_file

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

def run_single_ligand_prediction(config_file: Path, job_id: str, ligand_id: str, options: Dict[str, Any], 
                                device_id: int, results_dir: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """Run Boltz-2 prediction for a single ligand."""
    
    output_dir = results_dir / job_id / ligand_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up environment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(device_id)
    
    # Run boltz command
    cmd = [
        "boltz", "predict", str(config_file),
        "--use_msa_server",
        "--out_dir", str(output_dir),
        "--no_kernels",
        "--devices", "1",  # Use 1 device since we set CUDA_VISIBLE_DEVICES
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

def restructure_results_by_amide_worker(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Restructure results to map amide → ChEMBL ID → affinity data (worker version)."""
    amide_results = {}
    
    for pdb_result in results:
        if "amide_results" not in pdb_result:
            continue
            
        chembl_id = pdb_result.get("chembl_id", "unknown")
        pdb_file = pdb_result.get("pdb_file", "unknown")
        
        for ligand_id, ligand_result in pdb_result["amide_results"].items():
            if "error" in ligand_result:
                continue  # Skip failed predictions
                
            amide_name = ligand_result.get("amide_name", "unknown")
            amide_smiles = ligand_result.get("amide_smiles", "unknown")
            
            # Initialize amide entry if not exists
            if amide_name not in amide_results:
                amide_results[amide_name] = {
                    "amide_smiles": amide_smiles,
                    "chembl_results": {},
                    "total_predictions": 0,
                    "successful_predictions": 0
                }
            
            # Extract affinity data
            affinity_data = ligand_result.get("affinity_data", {})
            confidence_data = ligand_result.get("confidence_data", {})
            
            if affinity_data:
                # Calculate IC50 from affinity prediction
                pred_aff = affinity_data.get("affinity_pred_value")
                ic50_uM = 10**(pred_aff) * 10**3 if pred_aff is not None else None
                
                binding_prob = affinity_data.get("affinity_probability_binary")
                confidence_score = confidence_data.get("confidence_score") if confidence_data else None
                
                # Store structured result
                amide_results[amide_name]["chembl_results"][chembl_id] = {
                    "pdb_file": pdb_file,
                    "affinity_pred_value": pred_aff,
                    "ic50_uM": ic50_uM,
                    "binding_probability": binding_prob,
                    "confidence_score": confidence_score,
                    "duration": ligand_result.get("duration"),
                    "raw_affinity_data": affinity_data,
                    "raw_confidence_data": confidence_data
                }
                
                amide_results[amide_name]["successful_predictions"] += 1
            
            amide_results[amide_name]["total_predictions"] += 1
    
    return amide_results

def filter_pdbs_by_chembl_and_length(target_chembl_ids: List[str]) -> List[str]:
    """Filter PDBs by ChEMBL IDs and protein length."""
    # Load from app_data.json if available
    app_data_file = Path("app_data.json")
    if app_data_file.exists():
        with open(app_data_file, 'r') as f:
            app_data = json.load(f)
        orchard_config = app_data.get("orchard_config", {})
        pdb_pattern = orchard_config.get("paths", {}).get("pdb_files_pattern", "../pdb_files_by_chembl_id/*/*_clean.pdb")
        length_file = orchard_config.get("paths", {}).get("pdb_to_length_file", "./pdb_to_length.json")
        max_length = orchard_config.get("settings", {}).get("max_protein_length", 1000)
    else:
        # Fallback defaults
        pdb_pattern = "../pdb_files_by_chembl_id/*/*_clean.pdb"
        length_file = "./pdb_to_length.json"
        max_length = 1000
    
    all_pdbs = glob.glob(pdb_pattern)
    
    # Filter by ChEMBL ID if specified
    if target_chembl_ids:
        target_chembl_set = set(target_chembl_ids)
        filtered_pdbs = []
        for pdb_file in all_pdbs:
            # Extract ChEMBL ID from path: ../pdb_files_by_chembl_id/CHEMBLXXX/...
            chembl_id = pdb_file.split('/')[-2]
            if chembl_id in target_chembl_set:
                filtered_pdbs.append(pdb_file)
        all_pdbs = filtered_pdbs
    
    # Filter by protein length
    pdb_to_length = {}
    if os.path.exists(length_file):
        try:
            with open(length_file, 'r') as f:
                pdb_to_length = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load PDB length data: {e}")
    
    filtered_pdbs = []
    for pdb_file in all_pdbs:
        if pdb_file in pdb_to_length:
            length = pdb_to_length[pdb_file]
            if length < max_length:
                filtered_pdbs.append(pdb_file)
        else:
            # Include if no length data (conservative approach)
            filtered_pdbs.append(pdb_file)
    
    return filtered_pdbs

def main():
    if len(sys.argv) != 2:
        print("Usage: python slurm_boltz_worker.py <job_data.json>")
        sys.exit(1)
    
    job_data_file = sys.argv[1]
    
    # Load job data
    with open(job_data_file, 'r') as f:
        job_data = json.load(f)
    
    job_id = job_data["job_id"]
    amides_to_process = job_data["amides_to_process"]
    names_to_process = job_data["names_to_process"]
    target_chembl_ids = job_data["target_chembl_ids"]
    options = job_data["options"]
    num_gpus = job_data["num_gpus"]
    
    print(f"🚀 Starting SLURM worker for job {job_id}")
    print(f"   Amides: {len(amides_to_process)}")
    print(f"   GPUs: {num_gpus}")
    
    # Create directories
    timestamp = str(int(time.time()))
    pid = str(os.getpid())
    unique_id = f"{timestamp}_{pid}_{job_id}"
    
    results_dir = Path(f"/tmp/boltz2_results_{unique_id}")
    temp_dir = Path(f"/tmp/boltz2_temp_{unique_id}")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Filter PDBs
        filtered_pdbs = filter_pdbs_by_chembl_and_length(target_chembl_ids)
        print(f"   PDBs to process: {len(filtered_pdbs)}")
        
        if not filtered_pdbs:
            raise Exception("No PDBs found matching the criteria")
        
        # Set up GPU assignments
        available_gpus = list(range(num_gpus))
        
        # Process results
        all_results = []
        total_combinations = len(amides_to_process) * len(filtered_pdbs)
        processed_count = 0
        
        print(f"   Total combinations to process: {total_combinations}")
        
        # Process each PDB
        for pdb_idx, pdb_file in enumerate(filtered_pdbs):
            try:
                print(f"Processing PDB {pdb_idx + 1}/{len(filtered_pdbs)}: {os.path.basename(pdb_file)}")
                
                # Extract protein sequence
                protein_sequence = sequence_from_pdb(pdb_file)
                
                # Create config files for all amides
                config_files = []
                ligand_ids = []
                
                for i, (amide_smiles, amide_name) in enumerate(zip(amides_to_process, names_to_process)):
                    ligand_id = f"L{i+1}"
                    config_file = create_config_file(protein_sequence, amide_smiles, ligand_id, job_id, temp_dir)
                    config_files.append(config_file)
                    ligand_ids.append(ligand_id)
                
                # Run predictions in parallel for this PDB
                gpu_assignments = {ligand_id: available_gpus[i % num_gpus] for i, ligand_id in enumerate(ligand_ids)}
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
                    future_to_ligand = {
                        executor.submit(
                            run_single_ligand_prediction, 
                            config_file, 
                            job_id, 
                            ligand_id, 
                            options, 
                            gpu_assignments[ligand_id],
                            results_dir
                        ): (ligand_id, i) 
                        for i, (config_file, ligand_id) in enumerate(zip(config_files, ligand_ids))
                    }
                    
                    pdb_results = {}
                    for future in concurrent.futures.as_completed(future_to_ligand):
                        ligand_id, amide_idx = future_to_ligand[future]
                        try:
                            success, returned_ligand_id, result_data = future.result()
                            if success:
                                # Add metadata
                                result_data['amide_smiles'] = amides_to_process[amide_idx]
                                result_data['amide_name'] = names_to_process[amide_idx]
                                result_data['pdb_file'] = pdb_file
                                result_data['chembl_id'] = pdb_file.split('/')[-2] if '/' in pdb_file else 'unknown'
                                pdb_results[ligand_id] = result_data
                            else:
                                pdb_results[ligand_id] = {
                                    "error": result_data.get("error", "Unknown error"),
                                    "amide_smiles": amides_to_process[amide_idx],
                                    "amide_name": names_to_process[amide_idx],
                                    "pdb_file": pdb_file,
                                    "chembl_id": pdb_file.split('/')[-2] if '/' in pdb_file else 'unknown'
                                }
                            processed_count += 1
                        except Exception as e:
                            pdb_results[ligand_id] = {
                                "error": str(e),
                                "amide_smiles": amides_to_process[amide_idx],
                                "amide_name": names_to_process[amide_idx],
                                "pdb_file": pdb_file,
                                "chembl_id": pdb_file.split('/')[-2] if '/' in pdb_file else 'unknown'
                            }
                            processed_count += 1
                
                # Store results for this PDB
                pdb_result_entry = {
                    "pdb_file": pdb_file,
                    "chembl_id": pdb_file.split('/')[-2] if '/' in pdb_file else 'unknown',
                    "protein_sequence": protein_sequence,
                    "amide_results": pdb_results,
                    "processed_amides": len(amides_to_process)
                }
                all_results.append(pdb_result_entry)
                
                # Clean up config files for this PDB
                for config_file in config_files:
                    config_file.unlink(missing_ok=True)
                
                # Periodic cleanup
                if (pdb_idx + 1) % 20 == 0:
                    print(f"🧹 Performing periodic cleanup after {pdb_idx + 1} PDBs")
                    # Clean results but keep temp for ongoing work
                    if results_dir.exists():
                        for item in results_dir.iterdir():
                            if item.is_dir():
                                shutil.rmtree(item)
                
            except Exception as e:
                print(f"❌ Error processing PDB {pdb_file}: {e}")
                # Add error entry
                error_entry = {
                    "pdb_file": pdb_file,
                    "chembl_id": pdb_file.split('/')[-2] if '/' in pdb_file else 'unknown',
                    "error": str(e),
                    "processed_amides": 0
                }
                all_results.append(error_entry)
                continue
        
        # Compile final results
        end_time = time.time()
        
        final_results = {
            "job_id": job_id,
            "status": "completed",
            "summary": {
                "num_gpus_used": num_gpus,
                "amides_processed": len(amides_to_process),
                "amide_names": names_to_process,
                "pdbs_processed": len(filtered_pdbs),
                "total_combinations": total_combinations,
                "successful_predictions": processed_count,
                "target_chembl_ids": target_chembl_ids if target_chembl_ids else "all",
                "options": options
            },
            "results": all_results,
            "timestamp": time.time()
        }
        
        # Create structured results by amide
        structured_results = restructure_results_by_amide_worker(all_results)
        
        # Add structured results to final output
        final_results["structured_results"] = structured_results
        
        # Save results
        results_file = Path(f"./slurm_scripts/{job_id}_results.json")
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"✅ Job {job_id} completed successfully!")
        print(f"   Results saved to: {results_file}")
        print(f"   Processed {processed_count}/{total_combinations} combinations")
        
        # Print summary of results by amide
        print(f"📊 Results Summary by Amide:")
        for amide_name, amide_data in structured_results.items():
            successful = amide_data["successful_predictions"]
            total = amide_data["total_predictions"]
            print(f"   {amide_name}: {successful}/{total} successful predictions")
            
            # Show top 3 results by binding probability
            chembl_results = amide_data["chembl_results"]
            if chembl_results:
                sorted_results = sorted(
                    chembl_results.items(),
                    key=lambda x: x[1].get("binding_probability", 0),
                    reverse=True
                )[:3]
                
                for chembl_id, result in sorted_results:
                    bp = result.get("binding_probability", 0)
                    ic50 = result.get("ic50_uM", "N/A")
                    print(f"     {chembl_id}: Binding Prob={bp:.3f}, IC50={ic50} μM")
        
    except Exception as e:
        print(f"❌ Job {job_id} failed: {e}")
        # Save error results
        error_results = {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "timestamp": time.time()
        }
        
        results_file = Path(f"./slurm_scripts/{job_id}_results.json")
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(error_results, f, indent=2)
        
        sys.exit(1)
        
    finally:
        # Clean up directories
        try:
            if results_dir.exists():
                shutil.rmtree(results_dir)
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary directories")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up directories: {e}")

if __name__ == "__main__":
    main()
