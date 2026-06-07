from app.opensearch_client import client

info = client.info()

print(info)