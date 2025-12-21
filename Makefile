.PHONY: help build start up down clean logs

help:
	@echo "ETL Studio - Comandos disponibles:"
	@echo "  make start    - Construir e iniciar todos los servicios (recomendado)"
	@echo "  make build    - Solo construir todas las imágenes"
	@echo "  make up       - Solo iniciar servicios (sin reconstruir)"
	@echo "  make down     - Detener todos los servicios"
	@echo "  make clean    - Limpiar contenedores y volúmenes"
	@echo "  make logs     - Ver logs de todos los servicios"

build:
	@echo "Construyendo imagen base..."
	docker-compose build base
	@echo "Construyendo resto de servicios..."
	docker-compose build

start: build
	@echo "Iniciando servicios..."
	docker-compose up

up:
	@echo "Iniciando servicios (sin reconstruir)..."
	docker-compose up

down:
	docker-compose down

clean:
	docker-compose down -v
	docker rmi etl-studio-base:latest || true

logs:
	docker-compose logs -f
