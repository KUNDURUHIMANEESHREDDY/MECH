import sqlite3

for db in ['backend/storage/mech.db', 'backend/storage/artifacts.db', 'backend/storage/warm_model.db']:
    print(f"\n=== {db} ===")
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print('Tables:', tables)
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table[0]})")
            print(f'  {table[0]}:', cursor.fetchall())
    except Exception as e:
        print(f"  Error: {e}")