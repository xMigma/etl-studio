# Cleaning API (Silver Layer)

Endpoints para la limpieza de datos en la capa Silver.

## Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/cleaning/rules` | `GET` | Devuelve las reglas disponibles (fillna, trim, etc.) |
| `/cleaning/preview` | `POST` | Aplica las reglas y devuelve el resultado transformado para preview |
| `/cleaning/apply` | `POST` | Aplica las reglas y **guarda** el resultado en `data/silver/` |

---

## 1. GET /cleaning/rules

Devuelve las reglas de limpieza disponibles.

**Cuándo se llama:** Al cargar la página de cleaning.

### Request

```
GET /cleaning/rules
```

### Response

```json
{
  "fillna": {
    "name": "FillNA",
    "description": "Rellenar valores nulos",
    "requires_value": true
  },
  "trim": {
    "name": "Trim",
    "description": "Eliminar espacios en blanco",
    "requires_value": false
  },
  "lowercase": {
    "name": "Lowercase",
    "description": "Convertir a minúsculas",
    "requires_value": false
  },
  "cast_date": {
    "name": "Cast Date",
    "description": "Convertir a fecha",
    "requires_value": false
  }
}
```

---

## 2. POST /cleaning/preview

Aplica las reglas a los datos y devuelve el resultado transformado sin guardar.

**Cuándo se llama:** Cada vez que el usuario añade o elimina una regla.

### Request

```json
{
  "table": "customers",
  "rules": [
    {"rule_id": "trim", "column": "name", "value": ""},
    {"rule_id": "fillna", "column": "phone", "value": "N/A"}
  ]
}
```

### Response

```json
{
  "after": [
    {"name": "Juan", "phone": "N/A"},
    {"name": "Ana", "phone": "123"}
  ]
}
```

---

## 3. POST /cleaning/apply

Aplica las reglas y guarda el resultado en la capa Silver.

**Cuándo se llama:** Cuando el usuario pulsa "Guardar cambios".

### Request

```json
{
  "table": "customers",
  "rules": [
    {"rule_id": "trim", "column": "name", "value": ""},
    {"rule_id": "fillna", "column": "phone", "value": "N/A"}
  ]
}
```

### Response

```json
{
  "status": "ok",
  "rows": 150,
  "output_path": "data/silver/customers.csv"
}
```

---

## Flujo de uso

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Página carga ──────────► GET /cleaning/rules                │
│                               (obtiene reglas disponibles)       │
│                                                                  │
│  2. Usuario añade regla ───► POST /cleaning/preview             │
│     Usuario elimina regla ──► POST /cleaning/preview             │
│                               (actualiza preview cada vez)       │
│                                                                  │
│  3. Click "Guardar" ───────► POST /cleaning/apply               │
│                               (guarda en silver/)                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Reglas disponibles

| ID | Nombre | Descripción | Requiere valor |
|----|--------|-------------|----------------|
| `fillna` | FillNA | Rellenar valores nulos con un valor específico | ✅ |
| `trim` | Trim | Eliminar espacios en blanco al inicio y final | ❌ |
| `lowercase` | Lowercase | Convertir texto a minúsculas | ❌ |
| `cast_date` | Cast Date | Convertir columna a tipo fecha | ❌ |
