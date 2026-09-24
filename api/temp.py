from pathlib import Path

print(type(__file__))
print(type(Path(__file__)))

print(Path(__file__))
print(Path(__file__).resolve())