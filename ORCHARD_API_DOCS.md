# Boltz-2 Orchard Cluster API Documentation

This document describes the new `/predict_orchard` endpoint designed for running Boltz-2 predictions on the orchard cluster with specific GPU counts, amide filtering, and ChEMBL ID targeting.

## Endpoint: `/predict_orchard`

**Method:** POST  
**Content-Type:** application/json

### Request Parameters

#### Required Parameters

- **`num_gpus`** (integer): Number of GPUs to use for parallel processing. Must be a positive integer.
- **`amide_smiles`** (string or array): SMILES string(s) for the amide compounds to test. Can be:
  - Single string: `"O=C(NCCCCCCCC)C1=CC=CC=C1"`
  - Array of strings: `["O=C(NCCCCCCCC)C1=CC=CC=C1", "O=C(NC1=CC=CC=C1F)C2=CC=CC=C2"]`
- **`amide_names`** (string or array): Human-readable name(s) for the amide compounds. Must match the length of `amide_smiles`. Can be:
  - Single string: `"N-Octylbenzamide"`
  - Array of strings: `["N-Octylbenzamide", "N-(2-Fluorophenyl)benzamide"]`

#### Optional Parameters

- **`target_chembl_ids`** (array): List of ChEMBL IDs to filter PDB structures. If empty or not provided, all available PDBs will be processed.
  - Example: `["CHEMBL1075104", "CHEMBL1075105"]`
- **`diffusion_samples`** (integer): Number of diffusion samples. Default: 1
- **`recycling_steps`** (integer): Number of recycling steps. Default: 3  
- **`sampling_steps`** (integer): Number of sampling steps. Default: 200
- **`output_format`** (string): Output format ("pdb" or "mmcif"). Default: "mmcif"

### Example Request

```json
{
    "num_gpus": 2,
    "amide_smiles": [
        "O=C(NCCCCCCCC)C1=CC=CC=C1",
        "O=C(NC1=CC=CC=C1F)C2=CC=CC=C2"
    ],
    "amide_names": [
        "N-Octylbenzamide",
        "N-(2-Fluorophenyl)benzamide"
    ],
    "target_chembl_ids": [
        "CHEMBL1075104",
        "CHEMBL1075105",
        "CHEMBL1075106"
    ],
    "diffusion_samples": 1,
    "recycling_steps": 3,
    "sampling_steps": 200,
    "output_format": "mmcif"
}
```

### Response Format

#### Success Response (200)

```json
{
    "job_id": "abc12345",
    "status": "completed",
    "summary": {
        "num_gpus_used": 2,
        "amides_processed": 2,
        "amide_names": ["N-Octylbenzamide", "N-(2-Fluorophenyl)benzamide"],
        "pdbs_processed": 15,
        "total_combinations": 30,
        "successful_predictions": 28,
        "target_chembl_ids": ["CHEMBL1075104", "CHEMBL1075105", "CHEMBL1075106"],
        "options": {
            "diffusion_samples": 1,
            "recycling_steps": 3,
            "sampling_steps": 200,
            "output_format": "mmcif"
        }
    },
    "results": [
        {
            "pdb_file": "/path/to/pdb/file.pdb",
            "chembl_id": "CHEMBL1075104",
            "protein_sequence": "MVTPEGNVSLV...",
            "amide_results": {
                "L1": {
                    "affinity_data": {
                        "affinity_pred_value": -7.2,
                        "affinity_probability_binary": 0.85
                    },
                    "confidence_data": {
                        "confidence_score": 0.92
                    },
                    "structure_data": "...",
                    "duration": 120.5,
                    "amide_smiles": "O=C(NCCCCCCCC)C1=CC=CC=C1",
                    "amide_name": "N-Octylbenzamide",
                    "pdb_file": "/path/to/pdb/file.pdb",
                    "chembl_id": "CHEMBL1075104"
                },
                "L2": {
                    "affinity_data": {...},
                    "confidence_data": {...},
                    "amide_smiles": "O=C(NC1=CC=CC=C1F)C2=CC=CC=C2",
                    "amide_name": "N-(2-Fluorophenyl)benzamide",
                    "pdb_file": "/path/to/pdb/file.pdb",
                    "chembl_id": "CHEMBL1075104"
                }
            },
            "processed_amides": 2
        }
    ],
    "timestamp": 1703123456.789
}
```

#### Error Response (400/500)

```json
{
    "job_id": "abc12345",
    "status": "failed",
    "error": "Error description",
    "timestamp": 1703123456.789
}
```

## Configuration

The endpoint uses configuration from `app_data.json`:

```json
{
    "orchard_config": {
        "paths": {
            "amides_file": "./amides.txt",
            "pdb_to_length_file": "./pdb_to_length.json",
            "pdb_files_pattern": "../pdb_files_by_chembl_id/*/*_clean.pdb",
            "results_base_dir": "/project/flame/rmacknig/boltz2_configs",
            "predictions_base_dir": "/project/flame/rmacknig",
            "base_predictions_file": "/project/flame/rmacknig/predictions.json"
        },
        "settings": {
            "max_protein_length": 1000,
            "default_num_workers": 2,
            "cleanup_frequency": 20,
            "save_frequency": 5,
            "timeout_per_ligand": 1800,
            "cuda_visible_devices_env": true
        },
        "boltz_defaults": {
            "diffusion_samples": 1,
            "recycling_steps": 3,
            "sampling_steps": 200,
            "output_format": "mmcif",
            "diffusion_samples_affinity": "5",
            "max_parallel_samples": "1"
        }
    }
}
```

## Key Features

### 1. GPU Parallelization
- Distributes amide predictions across multiple GPUs
- Each GPU processes different ligands simultaneously
- Configurable GPU count via `num_gpus` parameter

### 2. ChEMBL ID Filtering
- Filter PDB structures by specific ChEMBL IDs
- Useful for targeting specific protein families or targets
- Empty list processes all available PDBs

### 3. Protein Length Filtering
- Automatically filters out proteins with >1000 residues (configurable)
- Prevents memory issues with very large proteins
- Uses `pdb_to_length.json` for efficient filtering

### 4. Automatic Cleanup
- Creates unique temporary directories per job
- Periodic cleanup during processing to manage disk space
- Final cleanup removes all temporary files

### 5. Error Handling
- Graceful handling of individual PDB failures
- Continues processing even if some predictions fail
- Detailed error reporting in results

## Usage Examples

### 1. Single Amide, All Targets
```python
data = {
    "num_gpus": 1,
    "amide_smiles": "O=C(NCCCCCCCC)C1=CC=CC=C1",
    "amide_names": "N-Octylbenzamide",
    "target_chembl_ids": []
}
```

### 2. Multiple Amides, Specific Targets
```python
data = {
    "num_gpus": 4,
    "amide_smiles": [
        "O=C(NCCCCCCCC)C1=CC=CC=C1",
        "O=C(NC1=CC=CC=C1F)C2=CC=CC=C2",
        "O=C(NC1=CC=C(CCCCCCC)C=C1)C2=CC=CC=C2"
    ],
    "amide_names": [
        "N-Octylbenzamide",
        "N-(2-Fluorophenyl)benzamide", 
        "N-(4-Heptylphenyl)benzamide"
    ],
    "target_chembl_ids": ["CHEMBL1075104", "CHEMBL1075105"]
}
```

### 3. High-Quality Predictions
```python
data = {
    "num_gpus": 2,
    "amide_smiles": "O=C(NCCCCCCCC)C1=CC=CC=C1",
    "amide_names": "N-Octylbenzamide",
    "target_chembl_ids": ["CHEMBL1075104"],
    "diffusion_samples": 5,
    "sampling_steps": 500,
    "recycling_steps": 5
}
```

## Performance Considerations

- **GPU Memory**: Each GPU processes one ligand at a time to avoid memory issues
- **Disk Space**: Temporary files are cleaned up periodically and after completion
- **Timeouts**: Default 30-minute timeout per ligand prediction
- **Parallelization**: Optimal performance with GPU count matching available hardware

## File Dependencies

The endpoint requires these files to be present:
- `app_data.json`: Configuration file
- `./pdb_to_length.json`: PDB length data for filtering
- PDB files matching the pattern in `pdb_files_pattern`

The endpoint will create/append to:
- `./amides.txt`: Amide SMILES and names database (created automatically from requests)

## Error Messages

Common error scenarios:
- `"num_gpus is required and must be a positive integer"`: Invalid or missing GPU count
- `"amide_smiles is required"`: Missing amide compounds
- `"amide_names is required"`: Missing amide names
- `"amide_smiles and amide_names must have the same length"`: Mismatched array lengths
- `"target_chembl_ids must be a list"`: Invalid ChEMBL ID format
- `"No PDBs found matching the criteria"`: No PDBs match the ChEMBL filter or length criteria
