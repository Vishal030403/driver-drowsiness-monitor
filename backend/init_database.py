from database import init_db, validate_database_setup

if __name__ == "__main__":
    print("Validating database setup...")
    validate_database_setup()
    
    print("Creating tables...")
    init_db()
    
    print("✅ Database initialized successfully!")