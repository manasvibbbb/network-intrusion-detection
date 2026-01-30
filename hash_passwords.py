# hash_passwords.py
import bcrypt

passwords = {
    'admin123': bcrypt.hashpw('admin123'.encode(), bcrypt.gensalt()).decode(),
    'viewer123': bcrypt.hashpw('viewer123'.encode(), bcrypt.gensalt()).decode()
}

print("Generated password hashes:")
for pwd, hash_val in passwords.items():
    print(f"{pwd}: {hash_val}")
