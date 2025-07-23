# Boltz-2 Binding Affinity Prediction API

A Flask API service for running Boltz-2 binding affinity predictions with support for both single and parallel multi-ligand processing.

## Features

- **Single Ligand Predictions**: Traditional protein-ligand binding affinity predictions
- **Multi-Ligand Parallel Processing**: Test multiple ligands against a single protein simultaneously
- **Automated Ligand Ranking**: Comparative analysis and ranking of multiple ligands
- **Job Management**: Track and retrieve results with unique job IDs
- **File Downloads**: Download structure files and prediction results
- **Real-time Monitoring**: Health check and job status endpoints

## Prerequisites

- Python 3.8+
- Boltz-2 installed and working (with `--no_kernels` flag support)
- CUDA environment properly configured

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure Boltz-2 is installed and working:
```bash
boltz predict --help
```

## Usage

### Start the API server:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

### API Endpoints

#### Health Check
```
GET /health
```

#### Single Ligand Prediction
```
POST /predict
Content-Type: application/json

{
    "protein_sequence": "MVTPEGNVSLVDESLLVGVTDEDRAV...",
    "ligand_smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
    "diffusion_samples": 1,       // optional, default 1, more = better quality
    "recycling_steps": 3,         // optional, default 3, more = better quality
    "sampling_steps": 200,        // optional, default 200, more = better quality
    "output_format": "mmcif"      // optional, "pdb" or "mmcif", default "mmcif"
}
```

#### Multi-Ligand Parallel Prediction ⚡ **NEW**
```
POST /predict_multi
Content-Type: application/json

{
    "protein_sequence": "MVTPEGNVSLVDESLLVGVTDEDRAV...",
    "ligands": [
        {
            "id": "Tyrosine",
            "smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
        },
        {
            "id": "Leucine", 
            "smiles": "CC(C)CC[C@H](N)C(=O)O"
        },
        {
            "id": "BOC-Lysine",
            "smiles": "CC(C)(C)OC(=O)N[C@@H](CCCCN)C(=O)O"
        }
    ],
    "diffusion_samples": 1,       // optional, applied to all ligands
    "recycling_steps": 3,         // optional, applied to all ligands
    "sampling_steps": 200,        // optional, applied to all ligands
    "output_format": "mmcif"      // optional, applied to all ligands
}
```

**Multi-Ligand Response Format:**
```json
{
    "job_id": "abc12345",
    "status": "completed",
    "summary": {
        "total_ligands": 3,
        "successful": 3,
        "failed": 0,
        "duration": 245.67,
        "successful_ligands": ["Tyrosine", "Leucine", "BOC-Lysine"],
        "failed_ligands": []
    },
    "results": {
        "Tyrosine": {
            "affinity_pred_value": -8.234,
            "affinity_probability_binary": 0.892,
            "iptm": 0.78,
            "ptm": 0.85,
            "duration": 82.34,
            "files": {
                "structure": ["tyrosine_model_0.cif"],
                "affinity": ["affinity_tyrosine.json"]
            }
        },
        "Leucine": { ... },
        "BOC-Lysine": { ... }
    },
    "analysis": {
        "ranking": [
            {
                "rank": 1,
                "ligand_id": "Tyrosine",
                "affinity_value": -8.234,
                "binding_probability": 0.892
            },
            {
                "rank": 2,
                "ligand_id": "BOC-Lysine",
                "affinity_value": -7.156,
                "binding_probability": 0.743
            },
            {
                "rank": 3,
                "ligand_id": "Leucine",
                "affinity_value": -6.432,
                "binding_probability": 0.621
            }
        ],
        "statistics": {
            "best_affinity": {
                "ligand_id": "Tyrosine",
                "value": -8.234
            },
            "worst_affinity": {
                "ligand_id": "Leucine",
                "value": -6.432
            },
            "mean_affinity": -7.274,
            "affinity_range": 1.802
        }
    }
}
```

#### Get Results
```
GET /results/{job_id}
```

#### List All Jobs
```
GET /jobs
```

#### Download Files
```
GET /download/{filename}
```

#### Clean Up Job
```
DELETE /cleanup/{job_id}
```

## API Features

### **Optional Parameters:**
- `diffusion_samples` (int): Number of diffusion samples (default: 1). Higher values give better results but take longer.
- `recycling_steps` (int): Number of recycling steps (default: 3). More steps improve quality but increase runtime.
- `sampling_steps` (int): Number of sampling steps (default: 200). Higher values improve quality but take longer.
- `output_format` (str): Output format for structure files, either "pdb" or "mmcif" (default: "mmcif").

### **Multi-Ligand Advantages:**
- **Parallel Processing**: Multiple ligands processed simultaneously instead of sequentially
- **Resource Efficiency**: Protein structure can be reused across ligands
- **Automated Ranking**: Ligands automatically ranked by binding affinity
- **Comparative Analysis**: Statistics and comparison metrics provided
- **Speedup**: Typical 2-4x faster than sequential processing

### **Ligand Input Formats:**
The multi-ligand endpoint accepts ligands in two formats:

**Format 1: Object with ID and SMILES**
```json
{
    "ligands": [
        {
            "id": "Compound_A",
            "smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
        }
    ]
}
```

**Format 2: SMILES strings only (auto-generates IDs)**
```json
{
    "ligands": [
        "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
        "CC(C)CC[C@H](N)C(=O)O"
    ]
}
```

## Testing

### Test the API endpoints:
```bash
# Make scripts executable
chmod +x test_api_multi_ligand.py

# Test multi-ligand endpoint
python test_api_multi_ligand.py
```

### Test parallel processing directly:
```bash
# Test the underlying parallel processing
python test_parallel_ligands.py
```

## Performance Considerations

- **Concurrent Jobs**: Limited to 4 parallel ligands by default to manage GPU memory
- **Timeout**: Each ligand prediction has a 30-minute timeout
- **Memory**: Monitor GPU memory usage with multiple concurrent predictions
- **Scaling**: For large-scale screening, consider multiple API instances

## Example Use Cases

1. **Lead Optimization**: Compare multiple variants of a lead compound
2. **Fragment Screening**: Screen small molecule fragments against a target
3. **SAR Analysis**: Structure-activity relationship studies
4. **Hit Ranking**: Rank potential hits from virtual screening
5. **Compound Series**: Evaluate a series of related compounds

## Network Configuration

The API binds to `0.0.0.0:5000` for external access. For production deployment:

1. Use a reverse proxy (nginx)
2. Configure SSL/TLS
3. Set up proper firewall rules
4. Consider using a job queue system (Redis/Celery) for scalability

## Error Handling

The API handles various error conditions:
- Invalid protein sequences or SMILES
- Boltz-2 prediction failures
- Timeout errors
- Resource exhaustion
- File system errors

Failed ligands in multi-ligand predictions are reported individually without affecting successful predictions.

## File Structure

```
boltz2_local/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── test_api_multi_ligand.py   # API testing script
├── test_parallel_ligands.py   # Direct parallel processing test
├── configs/                    # Configuration files
│   └── multi_ligand_batch/    # Individual ligand configs
├── results/                   # Prediction results
└── temp/                      # Temporary files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License. 