from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{
        "host": "localhost",
        "port": 9200
    }],
    http_compress=True,
    use_ssl=False,
    verify_certs=False
)

INDEX_NAME = "documents"


def create_index():

    if client.indices.exists(index=INDEX_NAME):
        print("Index already exists")
        return

    body = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {

                "content": {
                    "type": "text"
                },

                "source_file": {
                    "type": "keyword"
                },

                "embedding": {
                    "type": "knn_vector",
                    "dimension": 768
                }

            }
        }
    }

    client.indices.create(
        index=INDEX_NAME,
        body=body
    )

    print("Index created")