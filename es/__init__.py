from elasticsearch import Elasticsearch

class es(Elasticsearch):
    def __init__(self, url, db="csdn_db") -> None:
        self.db = db
        self.client = Elasticsearch(url)
    def search(self, keyword, top_k=0) -> list:
        self.query = {
            "query": {
                "multi_match": {
                    "analyzer": "ik_smart",  # 使用ik分词器进行分词处理
                    "query": keyword,
                    "fields": [
                        "title",
                        "subtitle"
                    ]
                }
            }    
        }
        response = self.client.search(index=self.db, body = self.query)
        res = []
        for hit in response["hits"]["hits"]:
            score = hit["_score"]
            doc_data = hit["_source"]
            res.append((score,doc_data))
        if top_k == 0:
            return res
        else:
            top_k = min(top_k, len(res))
            return res[:top_k]

    