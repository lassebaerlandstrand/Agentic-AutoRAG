import os

# Silence Lance's "No existing dataset ... it will be created" WARN. Fires on
# every fresh memory:// connection and every trial rebuild, drowning useful
# logs. LanceDB wraps env_logger with a custom env var (LANCEDB_LOG) rather
# than the default RUST_LOG, and it must be set before `lancedb` is imported —
# env_logger reads it exactly once at library init.
os.environ.setdefault("LANCEDB_LOG", "lance::dataset::write::insert=off")
