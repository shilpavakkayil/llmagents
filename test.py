import magic

# Test if python-magic can detect file type
file_path = "skills.txt"  # Replace with a file path
mime = magic.Magic(mime=True)
file_type = mime.from_file(file_path)
print(file_type)