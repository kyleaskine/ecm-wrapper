# ECM Distributed Factorization System

A distributed system for integer factorization using ECM, P-1, P+1, and other methods.

## Components

### Client (`client/`)
Standalone Python client that can be downloaded and run independently:
- **GMP-ECM wrapper** for elliptic curve factorization
- **YAFU wrapper** for multi-method factorization (ECM, P±1, QS, NFS)  
- **Batch processing scripts** for bulk factorization
- **Configurable API submission** to central server

### Server (`server/`)
FastAPI-based coordination server:
- **Work assignment** and deduplication
- **Results aggregation** and storage
- **Progress tracking** across distributed clients
- **PostgreSQL backend** for persistence

## Quick Start

### Client Only
```bash
cd client/
pip install requests pyyaml
python3 ecm-wrapper.py --composite "123456789012345" --curves 100
```

### Full System (Development)
```bash
docker-compose up
# Server runs on http://localhost:8000
# PostgreSQL on localhost:5432
```

## Documentation

- [`client/CLAUDE.md`](client/CLAUDE.md) - Client implementation details and usage
- [`CLAUDE.md`](CLAUDE.md) - Development commands and project overview

## Architecture

```
┌─────────────┐    HTTP/API    ┌─────────────┐
│   Client    │◄─────────────►│   Server    │
│   (GMP-ECM, │               │  (FastAPI,  │
│    YAFU)    │               │ PostgreSQL) │
└─────────────┘               └─────────────┘
```

Clients are completely autonomous - they can run offline and submit results when network is available.