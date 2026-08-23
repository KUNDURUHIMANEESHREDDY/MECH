from backend.storage.migrations import migrate, MIGRATIONS, backup_db, get_schema_version

# Test mech.db
print("=== Testing mech.db ===")
migrate('backend/storage/mech.db', MIGRATIONS['mech.db'])
print(f"Schema version: {get_schema_version('backend/storage/mech.db')}")

import sqlite3
c = sqlite3.connect('backend/storage/mech.db')
print('tables:', c.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall())

# Test artifacts.db
print("\n=== Testing artifacts.db ===")
migrate('backend/storage/artifacts.db', MIGRATIONS['artifacts.db'])
print(f"Schema version: {get_schema_version('backend/storage/artifacts.db')}")

c = sqlite3.connect('backend/storage/artifacts.db')
print('tables:', c.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall())

# Test warm_model.db
print("\n=== Testing warm_model.db ===")
migrate('backend/storage/warm_model.db', MIGRATIONS['warm_model.db'])
print(f"Schema version: {get_schema_version('backend/storage/warm_model.db')}")

c = sqlite3.connect('backend/storage/warm_model.db')
print('tables:', c.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall())

# Test backup
print("\n=== Testing backup ===")
backup_path = backup_db('backend/storage/warm_model.db')
print(f"Backup created: {backup_path}")