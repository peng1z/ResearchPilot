.PHONY: up down logs test test-backend test-frontend

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

test: test-backend test-frontend

test-backend:
	cd backend && PYTHONPYCACHEPREFIX=/tmp/researchpilot-pycache python3 -m pytest

test-frontend:
	cd frontend && npm test

