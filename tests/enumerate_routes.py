import urllib.request
import json

res = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/openapi.json").read())
routes = []
for path, methods in res.get("paths", {}).items():
    for method, detail in methods.items():
        routes.append({"method": method.upper(), "path": path, "summary": detail.get("summary", "")})
print(f"Total Registered OpenAPI Endpoints: {len(routes)}")
for r in sorted(routes, key=lambda x: (x["path"], x["method"])):
    print(f"{r['method']:<6} {r['path']:<40} {r['summary']}")
