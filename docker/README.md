# 🧲 docker

> [!NOTE]
> Each runtime/worker has a `MAGNET_ROLE` environment variable.

Current role abstractions:

- "index"
    - Stream data from a `field`, encode using an embedding model, and upload to an index.