"""
Database connection and configuration for User Management Service
"""

import os
from databases import Database


# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./user_management.db"
)

# Create database instance
database = Database(DATABASE_URL)