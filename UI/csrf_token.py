import secrets

# Generate a secure random secret key
secret_key = secrets.token_hex(16)  # Generates a 32-character (16 bytes) hexadecimal string

print(secret_key)  # Print the generated secret key
