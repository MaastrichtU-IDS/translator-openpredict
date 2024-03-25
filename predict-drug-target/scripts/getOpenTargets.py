from ftplib import FTP
import os

def download_ftp_files(host, username, password, remote_directory, local_directory):
    """
    Download all files from a remote FTP directory.
    Parameters:
    - host (str): FTP host
    - username (str): FTP username
    - password (str): FTP password
    - remote_directory (str): Path to the remote FTP directory
    - local_directory (str): Local directory to save the downloaded files
    """

    # Connect to the FTP server and login
    with FTP(host) as ftp:
        ftp.login(username, password)

        # Change to the remote directory
        ftp.cwd(remote_directory)

        # List all files in the remote directory
        filenames = ftp.nlst()

        for filename in filenames:
            # Ensure local directory exists
            if not os.path.exists(local_directory):
                os.makedirs(local_directory)

            # Download each file
            local_path = os.path.join(local_directory, filename)
            with open(local_path, 'wb') as local_file:
                ftp.retrbinary('RETR ' + filename, local_file.write)

        print(f"Downloaded {len(filenames)} files to {local_directory}.")

# Usage
# Replace the below placeholders with actual values
# download_ftp_files('ftp.example.com', 'username', 'password', '/path/on/ftp', './local/directory')

dir = 'http://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/json/knownDrugsAggregated/'
host = 'ftp.ebi.ac.uk'
username = 'anonymous'
password = 'blah@blah.com'
remote_directory = 'pub/databases/opentargets/platform/23.09/output/etl/json/knownDrugsAggregated/'
local_directory = 'data'
download_ftp_files(host, username, password, remote_directory, local_directory)
