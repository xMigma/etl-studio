# PostgreSQL Docker Setup

Este directorio contiene la configuración para levantar una base de datos PostgreSQL 15 usando Docker.

## Estructura

- `Dockerfile`: Imagen personalizada de PostgreSQL 15 con Python 3
- `init_db.py`: Script de inicialización que crea las tablas necesarias

## Tablas Creadas

El script `init_db.py` crea las siguientes tablas:

1. **users**: Usuarios del sistema
2. **etl_jobs**: Trabajos ETL ejecutados
3. **configurations**: Configuraciones del sistema
4. **etl_logs**: Logs de los trabajos ETL
5. **source_data**: Datos fuente para procesamiento

## Uso

Desde la raíz del proyecto, ejecuta:

```bash
docker-compose up -d
```

Para ver los logs:

```bash
docker-compose logs -f postgres
```

Para conectarte a la base de datos:

```bash
docker-compose exec postgres psql -U etl_user -d etl_database
```

Para detener el servicio:

```bash
docker-compose down
```

Para detener y eliminar los volúmenes (⚠️ esto borrará todos los datos):

```bash
docker-compose down -v
```

## Credenciales por Defecto

- **Usuario**: etl_user
- **Contraseña**: etl_password
- **Base de datos**: etl_database
- **Puerto**: 5432

## Conexión desde Python

```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="etl_user",
    password="etl_password",
    database="etl_database"
)
```
