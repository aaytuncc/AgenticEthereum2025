#!/usr/bin/env bash

set -e  # Exit on any error

REPO_PATH=$PWD

# Remove previous service build
echo "Checking for previous service build..."
if test -d learning_service; then
  echo "Removing previous service build (requires sudo permission)"
  sudo rm -r learning_service
fi

# Remove empty directories to avoid wrong hashes
echo "Removing empty directories..."
find . -empty -type d -delete

# Ensure that third-party packages are correctly synced
echo "Cleaning previous builds..."
make clean

echo "Fetching autonomy version..."
AUTONOMY_VERSION=v$(autonomy --version | awk '{print $3}')
echo "Autonomy version: $AUTONOMY_VERSION"

echo "Fetching AEA version..."
AEA_VERSION=v$(aea --version | awk '{print $3}')
echo "AEA version: $AEA_VERSION"

echo "Syncing autonomy packages..."
autonomy packages sync --source valory-xyz/open-aea:$AEA_VERSION --source valory-xyz/open-autonomy:$AUTONOMY_VERSION --update-packages

# Ensure hashes are updated
echo "Updating package hashes..."
autonomy packages lock

# Push packages to IPFS
echo "Pushing packages to IPFS..."
autonomy push-all

# Fetch the service
echo "Fetching the service..."
autonomy fetch --local --service valory/learning_service && cd learning_service

# Build the image
echo "Initializing autonomy and building the image..."
autonomy init --reset --author author --remote --ipfs --ipfs-node "/dns/registry.autonolas.tech/tcp/443/https"
autonomy build-image

# Copy .env file
echo "Copying .env file..."
cp $REPO_PATH/.env .

# Copy the keys and build the deployment
echo "Copying keys.json..."
cp $REPO_PATH/keys.json .


echo "Building the deployment..."
autonomy deploy build keys.json -ltm

# Adjust permissions on the created directories
echo "Adjusting permissions for abci_build..."
chmod -R 777 ./abci_build

# Run the deployment
echo "Running the deployment..."
autonomy deploy run --build-dir abci_build/
