# Ingest API (Bronze Layer)

Endpoints para la ingesta de datos en la capa Bronze.

## Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/bronze/tables` | `GET` | Devuelve la lista de tablas disponibles |
| `/bronze/tables/{name}` | `GET` | Devuelve el contenido CSV de una tabla |
| `/bronze/tables/{name}` | `DELETE` | Elimina una tabla |
| `/bronze/upload` | `POST` | Sube e ingesta archivos CSV |

---

## 1. GET /bronze/tables

Devuelve la lista de tablas disponibles en la capa Bronze.

**Cuándo se llama:** Al cargar la página de ingesta.

### Request

```
GET /bronze/tables
```

### Response

```json
[
  {"name": "customers", "rows": 150, "created_at": "2025-12-05T10:30:00Z"},
  {"name": "orders", "rows": 1200, "created_at": "2025-12-05T09:15:00Z"},
  {"name": "products", "rows": 85, "created_at": "2025-12-04T14:20:00Z"}
]
```

---

## 2. GET /bronze/tables/{name}

Devuelve el contenido de una tabla en formato CSV.

**Cuándo se llama:** Cuando el usuario pulsa "Ver" en una tabla.

### Request

```
GET /bronze/tables/customers
```

### Response

```csv
id,name,email,phone
1,Juan,juan@email.com,123456789
2,Ana,ana@email.com,987654321
```

**Content-Type:** `text/csv`

---

## 3. DELETE /bronze/tables/{name}

Elimina una tabla de la capa Bronze.

**Cuándo se llama:** Cuando el usuario pulsa "Eliminar" en una tabla.

### Request

```
DELETE /bronze/tables/customers
```

### Response

```json
{
  "status": "ok",
  "message": "Table 'customers' deleted successfully"
}
```

---

## 4. POST /bronze/upload

Sube uno o más archivos CSV y los ingesta en la capa Bronze.

**Cuándo se llama:** Cuando el usuario arrastra/selecciona archivos CSV.

### Request

```
POST /bronze/upload
Content-Type: multipart/form-data

files: [customers.csv, orders.csv]
```

### Response

```json
{
  "status": "ok",
  "uploaded": [
    {"name": "customers", "rows": 150},
    {"name": "orders", "rows": 1200}
  ]
}
```

---

## Flujo de uso

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Página carga ──────────► GET /bronze/tables                 │
│                               (lista tablas disponibles)         │
│                                                                  │
│  2. Click "Ver" ───────────► GET /bronze/tables/{name}          │
│                               (muestra contenido CSV)            │
│                                                                  │
│  3. Click "Eliminar" ──────► DELETE /bronze/tables/{name}       │
│                               (elimina tabla)                    │
│                                                                  │
│  4. Subir archivos ────────► POST /bronze/upload                │
│                               (ingesta CSVs en bronze/)          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Estructura de datos

### Tabla

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `name` | string | Nombre de la tabla (sin extensión) |
| `rows` | number | Número de filas en la tabla |
| `created_at` | string | Fecha de ingesta (ISO 8601) |

## Almacenamiento

Los archivos CSV se guardan en `data/bronze/{name}.csv`.
