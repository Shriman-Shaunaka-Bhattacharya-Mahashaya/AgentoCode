import urllib.request
import json

packages = [
    "groq", "mcp", "prompt_toolkit", "rich", "chromadb",
    "watchdog", "tree-sitter", "tree-sitter-python", "python-dotenv",
    "anyio", "pydantic"
]

results = {}
for pkg in packages:
    try:
        url = f"https://pypi.org/pypi/{pkg}/json"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results[pkg] = data["info"]["version"]
    except Exception as e:
        results[pkg] = str(e)

print(json.dumps(results, indent=2))
