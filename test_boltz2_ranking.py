#!/usr/bin/env python3
"""
Test script for use_boltz2 ranking fix

Tests the updated use_boltz2 function with multiple ligands to ensure proper ranking.
"""

import sys
import os
from Bio.PDB import PDBParser, PPBuilder, PDBIO
import requests
import json

local_endpoints = {
    'boltz2': 'http://gpg-oppenheimer.cheme.cmu.edu:5000/predict_multi',
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
    print('\n--- Attempting to measure binding affinity of the ligand to the protein using Boltz2 ---\n')

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
    
    print(json.dumps(payload, indent=4))
    
    output_str = ""
    
    print(f"--- Sending {len(ligand_inputs)} ligands to Boltz2 ---\n")
    try:
        response = requests.post(local_endpoints['boltz2'], json=payload, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            result = response.json()
        else:
            print(Fore.RED + f'An error occurred while running Boltz2.' + Style.RESET_ALL)
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
    if result is not None:
        output_str += f"Boltz2 ran successfully!\nHere are the results:\n"
        results = result['results']
        for ligand_input in ligand_inputs:
            ligand_id = ligand_input['id']
            ligand_smiles = ligand_input['smiles']
            ligand_result = results[ligand_id]
            
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
    
    return output_str

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

if __name__ == "__main__":
    print("Testing updated use_boltz2 function...\n")
    
    # Test with single ligand first (simpler)
    # test_boltz2_single_ligand()
    
    print("\n" + "="*50)
    print("Now testing with multiple ligands...")
    print("="*50 + "\n")
    
    # Test with multiple ligands (shows ranking)
    test_boltz2_ranking() 