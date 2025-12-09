#!/usr/bin/env python3
"""
SSH connection handler that supports SSH config files and ProxyCommand.
"""

import os
import re
import logging
import warnings
import paramiko
from pathlib import Path

# Aggressively suppress Paramiko logging and warnings
logging.getLogger("paramiko").setLevel(logging.CRITICAL)
logging.getLogger("paramiko.transport").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=ResourceWarning, module="paramiko")

# Disable paramiko's default logging to stderr
paramiko.util.log_to_file(os.devnull)


def remove_known_host_entry(hostname):
    """Remove entries for a hostname from known_hosts file."""
    known_hosts_path = Path.home() / '.ssh' / 'known_hosts'
    
    if not known_hosts_path.exists():
        return False
    
    try:
        # Read all lines
        with open(known_hosts_path, 'r') as f:
            lines = f.readlines()
        
        # Filter out lines containing the hostname
        filtered_lines = []
        removed_count = 0
        for line in lines:
            # Check if the line contains the hostname (could be at start or after a comma)
            if hostname in line or f"[{hostname}]" in line:
                removed_count += 1
                print(f"Removing known_hosts entry: {line.strip()[:50]}...")
            else:
                filtered_lines.append(line)
        
        # Write back the filtered lines
        if removed_count > 0:
            with open(known_hosts_path, 'w') as f:
                f.writelines(filtered_lines)
            print(f"✅ Removed {removed_count} entries for {hostname} from known_hosts")
            return True
        
        return False
    except Exception as e:
        print(f"Error modifying known_hosts: {e}")
        return False


def get_ssh_config_for_host(hostname):
    """Parse SSH config file and return configuration for a specific host."""
    ssh_config_path = Path.home() / '.ssh' / 'config'
    
    if not ssh_config_path.exists():
        return None
        
    ssh_config = paramiko.SSHConfig()
    with open(ssh_config_path, 'r') as f:
        ssh_config.parse(f)
    
    # Look up the host configuration
    host_config = ssh_config.lookup(hostname)
    
    return host_config


def create_ssh_client_with_config(hostname, username=None, key_filename=None, port=22, banner_timeout=15):
    """
    Create an SSH client that respects SSH config file settings.
    
    Args:
        hostname: The hostname to connect to (can be an alias from ssh config)
        username: Optional username (will use config value if not provided)
        key_filename: Optional key file (will use config value if not provided)
        port: Optional port (will use config value if not provided)
        banner_timeout: Timeout in seconds for reading SSH banner (default 15, increase for slow connections)
    
    Returns:
        paramiko.SSHClient: Connected SSH client
    """
    client = paramiko.SSHClient()
    # More permissive policy that auto-adds unknown hosts and updates changed hosts
    # This is less secure but necessary for cloud instances that may change
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # Try to load known_hosts, but don't fail if it doesn't exist
    try:
        client.load_system_host_keys()
    except:
        pass
    
    try:
        client.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
    except:
        pass
    
    # Get SSH config for this host
    host_config = get_ssh_config_for_host(hostname)
    
    if host_config:
        # Use values from SSH config, with function parameters as overrides
        actual_hostname = host_config.get('hostname', hostname)
        actual_username = username or host_config.get('user', os.getenv('USER'))
        actual_port = port if port != 22 else int(host_config.get('port', 22))
        
        # Handle identity file
        if not key_filename and 'identityfile' in host_config:
            # SSH config returns a list of identity files
            identity_files = host_config['identityfile']
            if identity_files:
                key_filename = os.path.expanduser(identity_files[0])
        
        # Handle ProxyCommand
        proxy_command = host_config.get('proxycommand')
        # Function to attempt connection with retry on host key failure
        def attempt_connection(retry_on_host_key_failure=True):
            try:
                if proxy_command:
                    # Replace %h with hostname and %p with port
                    proxy_cmd = proxy_command.replace('%h', actual_hostname)
                    proxy_cmd = proxy_cmd.replace('%p', str(actual_port))
                    
                    print(f"Using ProxyCommand: {proxy_cmd}")
                    
                    # Create a proxy using paramiko's ProxyCommand
                    proxy = paramiko.ProxyCommand(proxy_cmd)
                    
                    # Connect through the proxy
                    client.connect(
                        hostname=actual_hostname,
                        username=actual_username,
                        key_filename=key_filename,
                        port=actual_port,
                        sock=proxy,
                        timeout=30,
                        banner_timeout=banner_timeout
                    )
                else:
                    # Direct connection without proxy
                    client.connect(
                        hostname=actual_hostname,
                        username=actual_username,
                        key_filename=key_filename,
                        port=actual_port,
                        timeout=30,
                        banner_timeout=banner_timeout
                    )
            except paramiko.ssh_exception.SSHException as e:
                error_msg = str(e)
                # Check if this is a host key verification failure
                if "Host key for server" in error_msg and "does not match" in error_msg and retry_on_host_key_failure:
                    print(f"⚠️  Host key verification failed for {actual_hostname}")
                    print("Attempting to remove old host key and retry...")
                    
                    # Remove the conflicting host entry
                    if remove_known_host_entry(actual_hostname):
                        print("Retrying connection after removing old host key...")
                        # Retry connection without retry flag to avoid infinite loop
                        attempt_connection(retry_on_host_key_failure=False)
                    else:
                        raise
                else:
                    raise
        
        attempt_connection()
        
    else:
        # No SSH config found, use provided parameters
        try:
            client.connect(
                hostname=hostname,
                username=username,
                key_filename=key_filename,
                port=port,
                timeout=30,
                banner_timeout=banner_timeout
            )
        except paramiko.ssh_exception.SSHException as e:
            error_msg = str(e)
            # Check if this is a host key verification failure
            if "Host key for server" in error_msg and "does not match" in error_msg:
                print(f"⚠️  Host key verification failed for {hostname}")
                print("Attempting to remove old host key and retry...")
                
                # Remove the conflicting host entry
                if remove_known_host_entry(hostname):
                    print("Retrying connection after removing old host key...")
                    # Retry connection
                    client.connect(
                        hostname=hostname,
                        username=username,
                        key_filename=key_filename,
                        port=port,
                        timeout=30
                    )
                else:
                    raise
            else:
                raise
    
    return client


# Test function
if __name__ == "__main__":
    # Test reading SSH config
    print("Testing SSH config reader...")
    
    test_host = os.getenv('BOLTZ_2_SSH_HOST_ALIAS', 'your-ssh-host')
    
    host_config = get_ssh_config_for_host(test_host)
    if host_config:
        print(f"Found config for '{test_host}':")
        print(f"  Hostname: {host_config.get('hostname')}")
        print(f"  User: {host_config.get('user')}")
        print(f"  IdentityFile: {host_config.get('identityfile')}")
        print(f"  ProxyCommand: {host_config.get('proxycommand')}")
    else:
        print(f"No SSH config found for '{test_host}'")
        print("Set BOLTZ_2_SSH_HOST_ALIAS environment variable to your SSH config host alias")
    
    # Test connection
    if host_config:
        print(f"\nTesting connection to {test_host}...")
        try:
            client = create_ssh_client_with_config(test_host)
            print("✅ Successfully connected!")
            
            # Test a command
            stdin, stdout, stderr = client.exec_command('hostname && date')
            print(f"Remote output: {stdout.read().decode().strip()}")
            
            client.close()
        except Exception as e:
            print(f"❌ Connection failed: {e}")
