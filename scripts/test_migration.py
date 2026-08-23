from backend.storage.migrations import migrate, MIGRATIONS, backup_db

migrate('backend/storage/warm_model.db', MIGRATIONS['warm_model.db'])
print('Migration done')

import sqlite3
c = sqlite3.connect('backend/storage/warm_model.db')
print('schema_version:', c.execute('SELECT * FROM schema_version').fetchall())
print('tables:', c.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall())