#!/bin/bash

echo "📦 Applying migrations..."
make makemigrations
make migrate

echo "📤 Collecting static files..."
make collectstatic

echo "👤 Creating superuser..."
make createsuperuser

