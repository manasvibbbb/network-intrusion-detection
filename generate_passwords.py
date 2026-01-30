# generate_passwords.py
import bcrypt

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Generate hashes for your passwords
print("Admin password (admin123):", hash_password("admin123"))
print("Viewer password (viewer123):", hash_password("viewer123"))
