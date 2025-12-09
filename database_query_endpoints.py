#!/usr/bin/env python3
"""
Database query endpoints for the Boltz-2 API.
Provides endpoints to query existing predictions and trigger new calculations if needed.
"""

from flask import Blueprint, request, jsonify
from typing import Dict, List, Optional, Tuple
import os
import time
import logging
import warnings
from pathlib import Path

# Aggressively suppress Paramiko logging and warnings
logging.getLogger("paramiko").setLevel(logging.CRITICAL)
logging.getLogger("paramiko.transport").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=ResourceWarning, module="paramiko")

# Create blueprint for database query routes
db_query_bp = Blueprint('db_query', __name__)

def connect_to_remote_database(ssh_client, remote_db_path: str, conda_path: str = None, conda_env: str = None) -> Optional[Dict]:
    """Connect to the remote database and return connection info."""
    try:
        # Set default conda values if not provided
        if conda_path is None:
            conda_path = os.getenv('BOLTZ_2_CONDA_PATH', '/project/flame/rmacknig/miniconda3/etc/profile.d/conda.sh')
        if conda_env is None:
            conda_env = os.getenv('BOLTZ_2_CONDA_ENV', 'boltz2')
        
        # Check if database exists
        stdin, stdout, stderr = ssh_client.exec_command(f"test -f {remote_db_path} && echo 'exists'")
        result = stdout.read().decode().strip()
        
        if result != 'exists':
            return None
        
        # For now, just return that the database exists without trying to get stats
        # This avoids the conda/database_manager import issues
        return {
            "exists": True,
            "path": remote_db_path,
        }
        
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return None


def query_remote_database(ssh_client, remote_db_path: str, query_type: str, **kwargs) -> List[Dict]:
    """Query the remote database."""
    try:
        # Get conda configuration
        conda_path = kwargs.get('conda_path') or os.getenv('BOLTZ_2_CONDA_PATH', '/project/flame/rmacknig/miniconda3/etc/profile.d/conda.sh')
        conda_env = kwargs.get('conda_env') or os.getenv('BOLTZ_2_CONDA_ENV', 'boltz2')
        conda_setup = f"source {conda_path} && conda activate {conda_env} && "
        
        # Build the Python command to execute remotely
        if query_type == "molecule":
            molecule_name = kwargs.get('molecule_name', '')
            molecule_smiles = kwargs.get('molecule_smiles', '')
            protein_ids = kwargs.get('protein_ids', [])
            
            # Escape the query parameters properly
            molecule_name_escaped = molecule_name.replace("'", "\\'") if molecule_name else ''
            molecule_smiles_escaped = molecule_smiles.replace("'", "\\'") if molecule_smiles else ''
            
            cmd = f"""import json
from database_manager import Boltz2DatabaseManager

molecule_name = '{molecule_name_escaped}' if '{molecule_name_escaped}' else None
molecule_smiles = '{molecule_smiles_escaped}' if '{molecule_smiles_escaped}' else None
protein_ids = {protein_ids}

with Boltz2DatabaseManager('{remote_db_path}') as db:
    # Use OR logic: get predictions by name OR by SMILES
    predictions = []
    
    # First, try to get predictions by name
    if molecule_name:
        predictions_by_name = db.get_predictions(molecule_name=molecule_name)
        predictions.extend(predictions_by_name)
    
    # Then, try to get predictions by SMILES
    if molecule_smiles:
        predictions_by_smiles = db.get_predictions(molecule_smiles=molecule_smiles)
        # Add only if not already in the list (avoid duplicates)
        existing_ids = set((p.get('molecule_name', ''), p.get('molecule_smiles', ''), p.get('chembl_id', ''), p.get('pdb_id', '')) for p in predictions)
        for pred in predictions_by_smiles:
            pred_id = (pred.get('molecule_name', ''), pred.get('molecule_smiles', ''), pred.get('chembl_id', ''), pred.get('pdb_id', ''))
            if pred_id not in existing_ids:
                predictions.append(pred)
    
    # If neither name nor SMILES provided, get all predictions
    if not molecule_name and not molecule_smiles:
        predictions = db.get_predictions()
    
    # Filter by Protein IDs if provided (supports both ChEMBL IDs and uploaded PDB IDs)
    if protein_ids:
        # A protein_id could be stored in either chembl_id or pdb_id field depending on type
        predictions = [p for p in predictions if 
                      p.get('chembl_id') in protein_ids or 
                      p.get('pdb_id') in protein_ids]
    
    # Simplify the output - using values FROM THE DATABASE
    results = []
    for pred in predictions:
        results.append({{
            'molecule_name': pred.get('molecule_name', ''),
            'molecule_smiles': pred.get('molecule_smiles', ''),
            'chembl_id': pred.get('chembl_id', ''),
            'pdb_id': pred.get('pdb_id', ''),
            'protein_id': pred.get('chembl_id') or pred.get('pdb_id', ''),  # Combined field for API response
            'binding_probability': pred.get('binding_probability'),
            'confidence_score': pred.get('confidence_score'),
            'ptm': pred.get('ptm'),
            'iptm': pred.get('iptm'),
            'ligand_iptm': pred.get('ligand_iptm'),
            'status': pred.get('status', ''),
            'created_at': str(pred.get('created_at', ''))
        }})
    
print(json.dumps(results))"""
        
        elif query_type == "top_predictions":
            min_binding_prob = kwargs.get('min_binding_prob', 0.7)
            limit = kwargs.get('limit', 100)
            
            cmd = f"""import json
from database_manager import Boltz2DatabaseManager

with Boltz2DatabaseManager('{remote_db_path}') as db:
    predictions = db.get_predictions(min_binding_probability={min_binding_prob})
    predictions = predictions[:{limit}]
    
    results = []
    for pred in predictions:
        results.append({{
            'molecule_name': pred.get('molecule_name', ''),
            'molecule_smiles': pred.get('molecule_smiles', ''),
            'chembl_id': pred.get('chembl_id', ''),
            'pdb_id': pred.get('pdb_id', ''),
            'protein_id': pred.get('chembl_id') or pred.get('pdb_id', ''),  # Combined field for API response
            'binding_probability': pred.get('binding_probability'),
            'confidence_score': pred.get('confidence_score'),
            'ptm': pred.get('ptm'),
            'status': pred.get('status', ''),
            'created_at': str(pred.get('created_at', ''))
        }})
    
print(json.dumps(results))"""
        
        # Execute the command remotely
        full_cmd = f"{conda_setup}cd {os.path.dirname(remote_db_path)} && python -c \"{cmd}\""
        
        stdin, stdout, stderr = ssh_client.exec_command(full_cmd)
        
        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()
        
        if error:
            print(f"❌ Database query error: {error}")
            return []
        
        if output:
            import json
            try:
                results = json.loads(output)
                return results
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                return []
        
        return []
        
    except Exception as e:
        print(f"❌ Error querying database: {e}")
        return []

@db_query_bp.route('/molecule_results', methods=['POST'])
def get_molecule_results():
    """
    Query molecule-protein binding results from database.
    If not found, optionally submit calculation.
    
    Request:
    {
        "molecule_smiles": "...",
        "molecule_name": "...",  
        "protein_ids": ["CHEMBL123", "my_protein_123456.pdb"],  # Optional - if not provided, return all (supports ChEMBL IDs and uploaded PDB IDs)
        "submit_if_missing": true,  # Whether to submit jobs for missing results
        "wait_for_results": false   # Whether to wait for calculation to complete
    }
    """
    from app import ssh_cluster, ORCHARD_CONFIG, generate_job_id
    from ssh_config_handler import create_ssh_client_with_config
    
    data = request.json
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    molecule_smiles = data.get('molecule_smiles')
    molecule_name = data.get('molecule_name')
    # Support both protein_ids and chembl_ids for backward compatibility
    protein_ids = data.get('protein_ids') or data.get('chembl_ids', [])
    submit_if_missing = data.get('submit_if_missing', False)
    wait_for_results = data.get('wait_for_results', False)
    
    if not molecule_smiles and not molecule_name:
        return jsonify({"error": "Either molecule_smiles or molecule_name is required"}), 400
    
    # Get database configuration
    remote_work_dir = os.getenv('BOLTZ_2_REMOTE_WORK_DIR')
    if not remote_work_dir:
        remote_work_dir = ORCHARD_CONFIG.get("cluster_config", {}).get("remote_work_dir")
        if not remote_work_dir:
            return jsonify({"error": "BOLTZ_2_REMOTE_WORK_DIR not configured"}), 400
    
    remote_db_path = os.getenv('BOLTZ_2_DATABASE_PATH', f"{remote_work_dir}/boltz2_predictions.db")
    
    # Get SSH connection from pool
    ssh_client = None
    try:
        # Try to use connection pool first
        from flask import current_app
        ssh_pool = current_app.config.get('SSH_POOL')
        
        if ssh_pool:
            ssh_client = ssh_pool.get_connection()
        else:
            # Fallback to creating new connection
            ssh_host_alias = os.getenv('BOLTZ_2_SSH_HOST_ALIAS')
            if not ssh_host_alias:
                return jsonify({"error": "BOLTZ_2_SSH_HOST_ALIAS not configured"}), 400
            ssh_client = create_ssh_client_with_config(ssh_host_alias)
        
        # Check if database exists
        try:
            db_info = connect_to_remote_database(ssh_client, remote_db_path)
        except Exception as db_error:
            print(f"❌ Database connection check failed: {db_error}")
            return jsonify({
                "error": "Database connection check failed",
                "details": str(db_error),
            }), 500
        if not db_info:
            return jsonify({
                "status": "database_not_found",
                "message": "Prediction database not found on cluster",
                "database_path": remote_db_path
            }), 404
        
        # Query existing predictions
        try:
            existing_results = query_remote_database(
                ssh_client, 
                remote_db_path,
                "molecule",
                molecule_name=molecule_name,
                molecule_smiles=molecule_smiles,
                protein_ids=protein_ids,
                conda_path=os.getenv('BOLTZ_2_CONDA_PATH', '/project/flame/rmacknig/miniconda3/etc/profile.d/conda.sh'),
                conda_env=os.getenv('BOLTZ_2_CONDA_ENV', 'boltz2')
            )
        except Exception as query_error:
            print(f"❌ Database query failed: {query_error}")
            return jsonify({
                "error": "Database query failed",
                "details": str(query_error),
            }), 500
        
        # Analyze what we found vs what was requested
        found_protein_ids = set()
        results_by_protein = {}
        
        for result in existing_results:
            # Get the protein ID from either chembl_id or pdb_id field, or use the combined protein_id field
            protein_id = result.get('protein_id') or result.get('chembl_id') or result.get('pdb_id')
            if protein_id:
                found_protein_ids.add(protein_id)
                if protein_id not in results_by_protein:
                    results_by_protein[protein_id] = []
                results_by_protein[protein_id].append(result)
        
        # Determine missing Protein IDs
        if protein_ids:
            requested_protein_ids = set(protein_ids)
            missing_protein_ids = list(requested_protein_ids - found_protein_ids)
        else:
            missing_protein_ids = []
        
        # Perform analysis if results found
        analysis = {}
        if existing_results:
            # Calculate statistics
            binding_probs = [r.get('binding_probability', 0) or 0 for r in existing_results]
            confidence_scores = [r.get('confidence_score', 0) or 0 for r in existing_results]
            
            # Sort by binding probability for top predictions
            sorted_results = sorted(existing_results, 
                                  key=lambda x: x.get('binding_probability', 0) or 0, 
                                  reverse=True)
            
            # Distribution analysis
            distribution = {}
            ranges = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                     (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
            
            for low, high in ranges:
                range_key = f"{low:.1f}-{high:.1f}"
                distribution[range_key] = len([p for p in binding_probs if low <= p < high])
            
            analysis = {
                "statistics": {
                    "total_predictions": len(existing_results),
                    "unique_targets": len(found_protein_ids),
                    "high_confidence_count": len([p for p in binding_probs if p > 0.7]),
                    "very_high_confidence_count": len([p for p in binding_probs if p > 0.9]),
                    "average_binding_probability": sum(binding_probs) / len(binding_probs) if binding_probs else 0,
                    "max_binding_probability": max(binding_probs) if binding_probs else 0,
                    "min_binding_probability": min(binding_probs) if binding_probs else 0,
                    "average_confidence_score": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                },
                "distribution": distribution,
                "top_10_predictions": [
                    {
                        "rank": i + 1,
                        "chembl_id": pred.get('chembl_id', ''),
                        "pdb_id": pred.get('pdb_id', ''),
                        "protein_id": pred.get('protein_id', ''),
                        "binding_probability": pred.get('binding_probability', 0),
                        "confidence_score": pred.get('confidence_score', 0),
                        "ptm": pred.get('ptm', 0),
                        "is_high_confidence": (pred.get('binding_probability', 0) or 0) > 0.9
                    }
                    for i, pred in enumerate(sorted_results[:10])
                ],
                "best_prediction": {
                    "chembl_id": sorted_results[0].get('chembl_id', '') if sorted_results else None,
                    "pdb_id": sorted_results[0].get('pdb_id', '') if sorted_results else None,
                    "protein_id": sorted_results[0].get('protein_id', '') if sorted_results else None,
                    "binding_probability": sorted_results[0].get('binding_probability', 0) if sorted_results else None,
                    "confidence_score": sorted_results[0].get('confidence_score', 0) if sorted_results else None
                } if sorted_results else None
            }
        
        # Get the actual molecule name/SMILES from the database results
        db_molecule_name = None
        db_molecule_smiles = None
        if existing_results:
            # Use the first result to get the database values
            db_molecule_name = existing_results[0].get('molecule_name', '')
            db_molecule_smiles = existing_results[0].get('molecule_smiles', '')
        
        response_data = {
            "status": "found" if existing_results else "not_found",
            "molecule_name": db_molecule_name if db_molecule_name else molecule_name,
            "molecule_smiles": db_molecule_smiles if db_molecule_smiles else molecule_smiles,
            "query_parameters": {
                "searched_name": molecule_name,
                "searched_smiles": molecule_smiles,
                "requested_protein_ids": protein_ids
            },
            "analysis": analysis,
            "results": {
                "found_count": len(existing_results),
                "found_protein_ids": list(found_protein_ids),
                "missing_protein_ids": missing_protein_ids,
                "predictions": results_by_protein,
                # Keep old field names for backward compatibility
                "found_chembl_ids": list(found_protein_ids),  # Deprecated but maintained for compatibility
                "missing_chembl_ids": missing_protein_ids     # Deprecated but maintained for compatibility
            }
        }
        
        # Submit jobs for missing predictions if requested
        if submit_if_missing and missing_protein_ids and molecule_smiles:
            # Import the submission function
            from app import predict_cluster
            
            # Prepare submission data
            submission_data = {
                "molecule_smiles": molecule_smiles,
                "molecule_name": molecule_name or "Unknown",
                "protein_ids": missing_protein_ids,
                "submit_jobs": True
            }
            
            # Create a test request context for internal call
            from flask import Flask
            test_app = Flask(__name__)
            with test_app.test_request_context(json=submission_data):
                submission_response = predict_cluster()
                
                if hasattr(submission_response, 'json'):
                    submission_result = submission_response.json
                else:
                    submission_result = submission_response[0].json
                
                response_data["submission"] = {
                    "status": submission_result.get("status"),
                    "job_id": submission_result.get("job_id"),
                    "submitted_jobs": submission_result.get("submitted_job_ids", []),
                    "message": f"Submitted {len(submission_result.get('submitted_job_ids', []))} jobs for {len(missing_protein_ids)} missing Protein IDs"
                }
                
                # If wait_for_results is True, we could implement polling logic here
                # For now, just return the job information
                if wait_for_results:
                    response_data["submission"]["note"] = "Waiting for results not yet implemented. Use job_id to check status."
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500
    finally:
        # Return connection to pool or close it
        if ssh_client:
            from flask import current_app
            ssh_pool = current_app.config.get('SSH_POOL')
            if ssh_pool:
                ssh_pool.return_connection(ssh_client)
            else:
                try:
                    ssh_client.close()
                except:
                    pass

@db_query_bp.route('/top_predictions', methods=['GET'])
def get_top_predictions():
    """Get top predictions from the database."""
    from app import ORCHARD_CONFIG
    from ssh_config_handler import create_ssh_client_with_config
    
    min_binding_prob = request.args.get('min_binding_prob', 0.7, type=float)
    limit = request.args.get('limit', 100, type=int)
    
    # Get database configuration
    remote_work_dir = os.getenv('BOLTZ_2_REMOTE_WORK_DIR')
    if not remote_work_dir:
        remote_work_dir = ORCHARD_CONFIG.get("cluster_config", {}).get("remote_work_dir")
        if not remote_work_dir:
            return jsonify({"error": "BOLTZ_2_REMOTE_WORK_DIR not configured"}), 400
    
    remote_db_path = os.getenv('BOLTZ_2_DATABASE_PATH', f"{remote_work_dir}/boltz2_predictions.db")
    
    # Get SSH connection from pool
    ssh_client = None
    try:
        # Try to use connection pool first
        from flask import current_app
        ssh_pool = current_app.config.get('SSH_POOL')
        
        if ssh_pool:
            ssh_client = ssh_pool.get_connection()
        else:
            # Fallback to creating new connection
            ssh_host_alias = os.getenv('BOLTZ_2_SSH_HOST_ALIAS')
            if not ssh_host_alias:
                return jsonify({"error": "BOLTZ_2_SSH_HOST_ALIAS not configured"}), 400
            ssh_client = create_ssh_client_with_config(ssh_host_alias)
        
        # Query database
        predictions = query_remote_database(
            ssh_client,
            remote_db_path,
            "top_predictions",
            min_binding_prob=min_binding_prob,
            limit=limit
        )
        
        return jsonify({
            "status": "success",
            "min_binding_probability": min_binding_prob,
            "limit": limit,
            "count": len(predictions),
            "predictions": predictions
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500
    finally:
        # Return connection to pool or close it
        if ssh_client:
            from flask import current_app
            ssh_pool = current_app.config.get('SSH_POOL')
            if ssh_pool:
                ssh_pool.return_connection(ssh_client)
            else:
                try:
                    ssh_client.close()
                except:
                    pass


@db_query_bp.route('/database_stats', methods=['GET'])
def get_database_stats():
    """Get database statistics."""
    from app import ORCHARD_CONFIG
    from ssh_config_handler import create_ssh_client_with_config
    
    # Get database configuration
    remote_work_dir = os.getenv('BOLTZ_2_REMOTE_WORK_DIR')
    if not remote_work_dir:
        remote_work_dir = ORCHARD_CONFIG.get("cluster_config", {}).get("remote_work_dir")
        if not remote_work_dir:
            return jsonify({"error": "BOLTZ_2_REMOTE_WORK_DIR not configured"}), 400
    
    remote_db_path = os.getenv('BOLTZ_2_DATABASE_PATH', f"{remote_work_dir}/boltz2_predictions.db")
    
    # Get SSH connection from pool
    ssh_client = None
    try:
        # Try to use connection pool first
        from flask import current_app
        ssh_pool = current_app.config.get('SSH_POOL')
        
        if ssh_pool:
            ssh_client = ssh_pool.get_connection()
        else:
            # Fallback to creating new connection
            ssh_host_alias = os.getenv('BOLTZ_2_SSH_HOST_ALIAS')
            if not ssh_host_alias:
                return jsonify({"error": "BOLTZ_2_SSH_HOST_ALIAS not configured"}), 400
            ssh_client = create_ssh_client_with_config(ssh_host_alias)
        
        # Get database info
        db_info = connect_to_remote_database(ssh_client, remote_db_path)
        
        if db_info:
            return jsonify({
                "status": "success",
                "database_path": remote_db_path,
                "stats": db_info.get('stats', {})
            })
        else:
            return jsonify({
                "status": "database_not_found",
                "database_path": remote_db_path
            }), 404
            
    except Exception as e:
        return jsonify({
            "status": "error", 
            "error": str(e)
        }), 500
    finally:
        # Return connection to pool or close it
        if ssh_client:
            from flask import current_app
            ssh_pool = current_app.config.get('SSH_POOL')
            if ssh_pool:
                ssh_pool.return_connection(ssh_client)
            else:
                try:
                    ssh_client.close()
                except:
                    pass
