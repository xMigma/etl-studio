# Gold API (Integration Layer)

Endpoints para la integración de datos en la capa Gold mediante operaciones de join.

## Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/gold/join` | `POST` | Realiza un join entre dos tablas y devuelve el resultado |
| `/gold/tables` | `GET` | Devuelve la lista de tablas Gold guardadas |
| `/gold/tables` | `POST` | Guarda una tabla en la capa Gold |
| `/gold/tables/{name}` | `GET` | Devuelve el contenido CSV de una tabla Gold |
| `/gold/tables/{name}` | `DELETE` | Elimina una tabla Gold |

---

## 1. POST /gold/join

Realiza una operación de join entre dos tablas y devuelve el resultado.

**Cuándo se llama:** Cada vez que el usuario cambia la configuración del join (tablas, columnas clave o tipo de join).

### Request

```json
{
  "left_table": "customers",
  "right_table": "cities",
  "config": {
    "left_key": "city_id",
    "right_key": "city_id",
    "join_type": "inner"
  }
}
```

**Nota:** `left_table` y `right_table` son nombres de tablas existentes en la capa Silver. El backend es responsable de cargar los datos de estas tablas.

### Response

```json
{
  "result_table": [
    {"id": 1, "name": "Juan", "city_id": 10, "city_name": "Madrid"},
    {"id": 2, "name": "Ana", "city_id": 20, "city_name": "Barcelona"}
  ]
}
```

---

## 2. GET /gold/tables

Devuelve la lista de tablas disponibles en la capa Gold.

**Cuándo se llama:** Al cargar la página Gold y después de guardar una tabla.

### Request

```
GET /gold/tables
```

### Response

```json
[
  {"name": "customers_cities", "rows": 150, "created_at": "2025-12-11T10:30:00Z"},
  {"name": "orders_products", "rows": 1200, "created_at": "2025-12-11T09:15:00Z"},
  {"name": "sales_analytics", "rows": 850, "created_at": "2025-12-10T14:20:00Z"}
]
```

---

## 3. POST /gold/tables

Guarda una tabla en la capa Gold.

**Cuándo se llama:** Cuando el usuario pulsa "Guardar tabla Gold".

### Request

```json
{
  "name": "customers_cities",
  "data": [
    {"id": 1, "name": "Juan", "city_id": 10, "city_name": "Madrid"},
    {"id": 2, "name": "Ana", "city_id": 20, "city_name": "Barcelona"}
  ]
}
```

### Response

```json
{
  "status": "ok",
  "message": "Table 'customers_cities' saved successfully",
  "rows": 150,
  "output_path": "data/gold/customers_cities.csv"
}
```

---

## 4. GET /gold/tables/{name}

Devuelve el contenido de una tabla Gold en formato CSV.

**Cuándo se llama:** Cuando el usuario pulsa "Ver" en una tabla Gold.

### Request

```
GET /gold/tables/customers_cities
```

### Response

```csv
id,name,city_id,city_name
1,Juan,10,Madrid
2,Ana,20,Barcelona
```

**Content-Type:** `text/csv`

---

## 5. DELETE /gold/tables/{name}

Elimina una tabla de la capa Gold.

**Cuándo se llama:** Cuando el usuario pulsa "Eliminar" en una tabla Gold.

### Request

```
DELETE /gold/tables/customers_cities
```

### Response

```json
{
  "status": "ok",
  "message": "Table 'customers_cities' deleted successfully"
}
```

---

## Flujo de uso

```
────────────────────────────────────────────────────────────────────
                        FRONTEND
────────────────────────────────────────────────────────────────────

  1. Página carga ─────────► GET /gold/tables
                              (lista tablas Gold guardadas)

  2. Usuario configura join
     - Selecciona tabla izq.
     - Selecciona tabla der.
     - Selecciona columnas clave
     - Selecciona tipo de join
                           │
                           └─────────► POST /gold/join
                                       (preview del resultado)

  3. Click "Guardar" ──────► POST /gold/tables
                              (guarda en gold/)

  4. Click "Ver" ──────────► GET /gold/tables/{name}
                              (muestra contenido CSV)

  5. Click "Eliminar" ─────► DELETE /gold/tables/{name}
                              (elimina tabla)

────────────────────────────────────────────────────────────────────
```

## Tipos de Join disponibles

| Tipo | Descripción |
|------|-------------|
| `inner` | Devuelve solo las filas que tienen coincidencias en ambas tablas |
| `left` | Devuelve todas las filas de la tabla izquierda y las coincidencias de la derecha |
| `right` | Devuelve todas las filas de la tabla derecha y las coincidencias de la izquierda |
| `outer` | Devuelve todas las filas de ambas tablas, con NULL donde no hay coincidencias |

## Estructura de datos

### Join Config

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `left_key` | string | Nombre de la columna clave en la tabla izquierda |
| `right_key` | string | Nombre de la columna clave en la tabla derecha |
| `join_type` | string | Tipo de join a realizar (inner, left, right, outer) |

### Tabla Gold

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `name` | string | Nombre de la tabla (sin extensión) |
| `rows` | number | Número de filas en la tabla |
| `created_at` | string | Fecha de creación (ISO 8601) |

## Almacenamiento

Los archivos CSV se guardan en `data/gold/{name}.csv`.

## Notas de implementación

- Las tablas de entrada (`left_table` y `right_table`) se envían como nombres de tabla (strings)
- El backend debe cargar estas tablas desde la capa Silver usando los nombres proporcionados
- El resultado del join se devuelve como un array de objetos JSON
- Si la API no está disponible, el frontend usa `apply_mock_join` como fallback (que requiere los DataFrames completos)
- Las tablas fuente provienen de la capa Silver
- Los nombres de tabla por defecto siguen el patrón `{left_table}_{right_table}`
- **Ventaja:** Este enfoque reduce significativamente el tamaño del payload y mejora el rendimiento
