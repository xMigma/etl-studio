#!/usr/bin/env python3
"""
Initialization script for PostgreSQL database
"""

import os
import sys
import time
import psycopg2


def wait_for_postgres(host, user, password, database, max_retries=30):
    """Wait for PostgreSQL to be ready."""
    retries = 0
    while retries < max_retries:
        try:
            conn = psycopg2.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            conn.close()
            print("PostgreSQL listo")
            return True
        except psycopg2.OperationalError:
            retries += 1
            print(f"Esperando a PostgreSQL... ({retries}/{max_retries})")
            time.sleep(1)
    
    print("No se pudo conectar a PostgreSQL")
    return False


def create_schemas(conn):
    """Create bronze, silver and gold schemas."""
    cursor = conn.cursor()
    
    try:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS bronze;")
        print("Created schema 'bronze'")
        
        cursor.execute("CREATE SCHEMA IF NOT EXISTS silver;")
        print("Created schema 'silver'")
        
        cursor.execute("CREATE SCHEMA IF NOT EXISTS gold;")
        print("Created schema 'gold'")
        
        conn.commit()
        print("\nDatabase initialized successfully")
        
    except Exception as e:
        conn.rollback()
        print(f"\nError creating schemas: {e}")
        raise
    finally:
        cursor.close()

def create_mlflow_database(host, user, password, db_name):
    """Create MLflow database."""
    try:
        # Connect to default postgres database to create mlflow db
        conn = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            database='postgres'
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_name,)
        )
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f'CREATE DATABASE {db_name}')
            print(f"Created database '{db_name}'")
        else:
            print(f"Database '{db_name}' already exists")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error creating MLflow database: {e}")
        raise


def main():
    db_host = os.getenv('POSTGRES_HOST', 'localhost')
    db_user = os.getenv('POSTGRES_USER', 'etl_user')
    db_password = os.getenv('POSTGRES_PASSWORD', 'etl_password')
    db_name = os.getenv('POSTGRES_DB', 'etl_studio')
    mlflow_db = os.getenv('MLFLOW_DB', 'mlflow_db')

    print(f"Conectando a PostgreSQL en {db_host}...")
    print(f"Base de datos: {db_name}")
    print(f"Usuario: {db_user}")
    
    if not wait_for_postgres(db_host, db_user, db_password, db_name):
        sys.exit(1)
    
    try:
        conn = psycopg2.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        
        create_schemas(conn)
        conn.close()
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
