def create_tables():
    """Create database tables and perform migrations if needed."""
    db.create_all()
    
    try:
        # Check if we need to perform database migrations
        with db.engine.connect() as con:
            # Check User table for profile_picture and avatar_choice columns
            user_columns = con.execute(text("PRAGMA table_info(user)")).fetchall()
            existing_columns = [col[1] for col in user_columns]
            
            missing_columns = []
            if 'profile_picture' not in existing_columns:
                missing_columns.append('profile_picture')
            if 'avatar_choice' not in existing_columns:
                missing_columns.append('avatar_choice')
                
            if missing_columns:
                print(f"Adding new columns to User table: {', '.join(missing_columns)}")
                
                # Create new columns
                if 'profile_picture' in missing_columns:
                    db.session.execute(text('ALTER TABLE user ADD COLUMN profile_picture VARCHAR(255)'))
                if 'avatar_choice' in missing_columns:
                    db.session.execute(text('ALTER TABLE user ADD COLUMN avatar_choice INTEGER DEFAULT 0'))
                
                db.session.commit()
                print("User table migration complete.")
            
            # Check Assessment table for assessment_hour column
            assessment_columns = con.execute(text("PRAGMA table_info(assessment)")).fetchall()
            existing_columns = [col[1] for col in assessment_columns]
            
            if 'assessment_hour' not in existing_columns:
                print("Adding assessment_hour column to Assessment table")
                db.session.execute(text('ALTER TABLE assessment ADD COLUMN assessment_hour INTEGER DEFAULT 0'))
                db.session.commit()
                print("Assessment table migration complete.")
    except Exception as e:
        print(f"Error during database migration: {str(e)}")
        print("Falling back to standard table creation.")
        db.create_all()
        print("Database tables created successfully") 