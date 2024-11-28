#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[1]:


from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, GPTVectorStoreIndex, StorageContext
from llama_index.core.node_parser import SimpleNodeParser


# In[2]:


api_key = "2oFBnOvJ7eWirmHk0m3iUVgY5CQ7d2Iv"
llm = MistralAI(api_key=api_key,model="mistral-large-latest")
embed_model = MistralAIEmbedding(model_name='mistral-embed', api_key=api_key)

Settings.llm = llm
Settings.embed_model = embed_model


# In[3]:


# Загрузка данных
reader = SimpleDirectoryReader(input_dir="books")
documents = reader.load_data()


# In[4]:


# Извлечение узлов
node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
nodes = node_parser.get_nodes_from_documents(documents)


# In[5]:


# разбитие на чанки и шифрование в эмбеддинги
index = GPTVectorStoreIndex(nodes)


# In[6]:


# созданеие файла с чанками
index.storage_context.persist('chunks')


# In[ ]:




