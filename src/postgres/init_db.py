#!/usr/bin/env python3
"""
Script de inicialización de base de datos PostgreSQL
Crea las tablas necesarias para el proyecto ETL Studio
"""

import os
import sys
import time
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def wait_for_postgres(host, user, password, database, max_retries=30):
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


def create_tables(conn):
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("Tabla 'users' creada")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etl_jobs (
                id SERIAL PRIMARY KEY,
                job_name VARCHAR(255) NOT NULL,
                job_type VARCHAR(50) NOT NULL,
                status VARCHAR(50) DEFAULT 'pending',
                created_by INTEGER REFERENCES users(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT
            );
        """)
        print("Tabla 'etl_jobs' creada")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS configurations (
                id SERIAL PRIMARY KEY,
                config_key VARCHAR(255) UNIQUE NOT NULL,
                config_value TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("Tabla 'configurations' creada")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etl_logs (
                id SERIAL PRIMARY KEY,
                job_id INTEGER REFERENCES etl_jobs(id) ON DELETE CASCADE,
                log_level VARCHAR(20) NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("Tabla 'etl_logs' creada")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS source_data (
                id SERIAL PRIMARY KEY,
                source_name VARCHAR(255) NOT NULL,
                data_type VARCHAR(100),
                raw_data JSONB,
                processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("Tabla 'source_data' creada")
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_etl_jobs_status 
            ON etl_jobs(status);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_etl_logs_job_id 
            ON etl_logs(job_id);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_data_processed 
            ON source_data(processed);
        """)
        
        print("Índices creados")
        
        cursor.execute("""
            INSERT INTO users (username, email) 
            VALUES 
                ('admin', 'admin@etl-studio.com'),
                ('etl_user', 'user@etl-studio.com')
            ON CONFLICT (username) DO NOTHING;
        """)
        
        cursor.execute("""
            INSERT INTO configurations (config_key, config_value, description)
            VALUES
                ('max_concurrent_jobs', '5', 'Número máximo de jobs ETL concurrentes'),
                ('default_timeout', '3600', 'Timeout por defecto en segundos'),
                ('log_retention_days', '30', 'Días de retención de logs')
            ON CONFLICT (config_key) DO NOTHING;
        """)
        
        print("Datos de ejemplo insertados")
        
        conn.commit()
        print("\nBase de datos inicializada correctamente")
        
    except Exception as e:
        conn.rollback()
        print(f"\nError al crear tablas: {e}")
        raise
    finally:
        cursor.close()


def main():
    db_host = os.getenv('POSTGRES_HOST', 'localhost')
    db_user = os.getenv('POSTGRES_USER', 'etl_user')
    db_password = os.getenv('POSTGRES_PASSWORD', 'etl_password')
    db_name = os.getenv('POSTGRES_DB', 'etl_database')
    
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
        
        create_tables(conn)
        conn.close()
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
