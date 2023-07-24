from elasticsearch import Elasticsearch

# 创建索引
def create_index(index_name, es):
    index_body = {
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "text", 
                    "analyzer": "ik_max_word", # 使用ik分词器进行分词处理
                    "search_analyzer": "ik_smart"
                },
                "subtitle": {
                    "type": "text",
                    "analyzer": "ik_max_word", # 使用ik分词器进行分词处理
                    "search_analyzer": "ik_smart"
                }
            }
        }
    }
    es.indices.create(index = index_name, body = index_body)
    print(f"Index '{index_name}' created successfully.")

if __name__ == "__main__":
    # 创建Elasticsearch客户端
    es = Elasticsearch("http://127.0.0.1:9200")

    # 测试连接
    if es.ping():
        print("Elasticsearch连接成功")
    else:
        print("Elasticsearch连接失败")
    index_name ="csdn"
    # 创建索引
    create_index(index_name,es)

    # 上传文档
    # doc = {
    #         "title": "测试title",
    #         "subtitle": "测试subtitle",
    #         "text": ["这是一段文本", "这是一段代码", "这是一段测试文本"],
    #         "code": [1],
    # }
    # es.index(index=index_name, document=doc)