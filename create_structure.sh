#!/bin/bash

# Create main directories
mkdir -p api/routes
mkdir -p chains
mkdir -p indexing
mkdir -p retrieval
mkdir -p augmentation
mkdir -p generation
mkdir -p config
mkdir -p utils
mkdir -p chrome-extension/popup
mkdir -p chrome-extension/content
mkdir -p chrome-extension/utils
mkdir -p tests
mkdir -p logs
mkdir -p docs

# Create __init__.py files
touch api/__init__.py
touch api/routes/__init__.py
touch chains/__init__.py
touch indexing/__init__.py
touch retrieval/__init__.py
touch augmentation/__init__.py
touch generation/__init__.py
touch config/__init__.py
touch utils/__init__.py
touch tests/__init__.py

# Create .gitkeep for empty directories
touch logs/.gitkeep
touch docs/.gitkeep

echo "Directory structure created successfully"