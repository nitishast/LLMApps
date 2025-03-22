# %%
import os
import shutil
import pytesseract
from pdf2image import convert_from_path
import traceback

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Read the input text file
file_path = "extracted_text.txt"
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Step 2: Process the text into LangChain Document format
documents = []
current_page = None
line_number = 0

for line in lines:
    line = line.strip()  # Remove leading/trailing whitespace
    if line.startswith("--- Page"):  # Detect page headers
        # Extract the page number from the header (e.g., '--- Page 1 ---')
        try:
            current_page = int(line.split()[-2])  # Get the page number (penultimate element)
        except ValueError:
            continue  # Skip if the page number is invalid (just in case)
        line_number = 0  # Reset line number for a new page
    elif line:  # Non-empty line
        documents.append(
            Document(
                page_content=line,
                metadata={"page": current_page, "line_number": line_number}
            )
        )
        line_number += 1

# Step 3: Split the documents using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=70,
    length_function=len,
    is_separator_regex=False,
)

final_documents = text_splitter.split_documents(documents)

# Check the result
for doc in final_documents[:5]:  # Print the first 5 chunks for inspection
    print(doc)



# %%
import fasttext as ft

# # # Download the FastText model
# !wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hi.zip
# !unzip wiki.hi.zip

# Load the FastText model
embedding_model_path = '/Users/nitastha/Desktop/NitishFiles/Projects/wiki.hi/wiki.hi.bin'
embed_model = ft.load_model(embedding_model_path)

# %%
import pandas as pd

# convert the documents to a dataframe
# This dataframe will be used to create the embeddings
# And later will be used to update the Qdrant Vector Database
docs = final_documents
data = []
for doc in docs:
   # Get the page content and metadata for each chunk
   # Meta data contains chunk source or file name
   row_data = {
       "page_content": doc.page_content,
       "metadata": doc.metadata
   }
   data.append(row_data)

df = pd.DataFrame(data)

# Replace the new line characters with space
df['page_content'] = df['page_content'].replace('\\n', ' ', regex=True)

# Create a unique id for each document.
# This id will be used to update the Qdrant Vector Database
df['id'] = range(1, len(df) + 1)

# Create a payload column in the dataframe
# This payload column includes the page content and metadata
# This payload will be used when LLM needs to answer a query
df['payload'] = df[['page_content', 'metadata']].to_dict(orient='records')

# Create embeddings for each chunk
# This embeddings will be used when doing a similarity search with the user query
df['embeddings'] = df['page_content'].apply(lambda x: (embed_model.get_sentence_vector(x)).tolist())


# %%
df.head()

# %%
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Batch

# Create a QdrantClient object
host = 'localhost'
port = 6333
client = QdrantClient(host=host, port=port)


# %%


# %%


# %%
# delete the collection if it already exists
# client.delete_collection(collection_name="my_collection")

# Create a fresh collection in Qdrant
client.create_collection(
  collection_name="my_collection",
  vectors_config=VectorParams(size=300, distance=Distance.COSINE),
)

# Update the Qdrant Vector Database with the embeddings
# We are updating the embeddings in batches
# Since the data is large, we will only update the first batch of size 4000
batch_size = 4000
client.upsert(
collection_name="my_collection",
points=Batch(
    ids=df['id'].to_list()[:batch_size],
    payloads=df['payload'][:batch_size],
    vectors=df['embeddings'].to_list()[:batch_size],
),
)

# Close the QdrantClient
# client.close()

# %%
import mlflow
from qdrant_client import QdrantClient
mlflow.end_run()
mlflow_lgging = True

if mlflow_lgging:
   # set the experiment name in the mlflow
   mlflow.set_experiment("Hindi Chatbot")
   # start the mlflow run
   mlflow.start_run()

# load the Qdrant client from the same host and port
# this client will be used to interact with the Qdrant server
host = "localhost"
port = 6333
client = QdrantClient(host=host, port=port)

# log the parameters in the mlflow
if mlflow_lgging:
   mlflow.log_param("qdrant_host", host)
   mlflow.log_param("qdrant_port", port)

# %%
if mlflow_lgging:
   mlflow.log_param("embed_model_path", embedding_model_path)

# %%
from typing import List
from qdrant_client import QdrantClient
import fasttext as ft
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Define a custom retriever class that uses Qdrant for document retrieval
# Since we're using FastText embeddings, we won't be able to use the default lanchain retriever, as it only supports HuggingFace and OpenAI Models
class QdrantRetriever(BaseRetriever):
   client: QdrantClient
   embed_model: ft.FastText._FastText
   collection_name: str
   limit: int

   def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
       """Converts query to a vector and retrieves relevant documents using Qdrant."""
       # Get the vector representation of the query using the FastText model
       query_vector = self.embed_model.get_sentence_vector(query).tolist()

       # Search for the most similar documents in the Qdrant collection
       # The search method returns a list of hits, where each hit contains the most similar document
       # we can limit the number of hits to return using the limit parameter
       search_results = self.client.search(
           collection_name=self.collection_name,
           query_vector=query_vector,
           limit=self.limit
       )
       # Finally, we convert the search results to a list of Document objects
       # that can be used by the pipeline
       return [Document(page_content=hit.payload['page_content']) for hit in search_results]

collection_name="my_collection"
limit = 500

# use the Custom QdrantRetriever class to create a retriever object
retriever = QdrantRetriever(
   client=client,
   embed_model=embed_model,
   collection_name=collection_name,
   limit=limit
)

if mlflow_lgging:
   mlflow.log_param("collection_name", collection_name)
   mlflow.log_param("limit", limit)

# %%
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ### gemini api

# %%
from langchain_google_genai import ChatGoogleGenerativeAI
import config


# Set up Gemini API
model_name = config.CHAT_MODEL  # Gemini Pro model
google_api_key = config.GOOGLE_API_KEY  # Replace with your actual Google API key

llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=google_api_key,
    temperature=0.6,
    max_output_tokens=500
)

from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    """<s>[INST] आप एक विश्वसनीय और सटीक सहायक हैं। आपको केवल और केवल नीचे दिए गए संदर्भ के आधार पर प्रश्न का उत्तर देना है। 

निर्देश:
- केवल दिए गए संदर्भ से जानकारी का उपयोग करें
- यदि संदर्भ में उत्तर नहीं मिलता है, तो स्पष्ट रूप से कहें कि "दिए गए संदर्भ में इस प्रश्न का उत्तर नहीं मिलता"
- अपने ज्ञान या अतिरिक्त जानकारी को शामिल न करें
- उत्तर संक्षिप्त, स्पष्ट और सीधा होना चाहिए
- हिंदी भाषा में ही उत्तर दें

संदर्भ: {context} </s>
"""
)

prompt = ChatPromptTemplate.from_messages(
   [
       ("system", system_prompt),
       ("human", "{input}"),
   ]
)

# if mlflow_lgging:
#    mlflow.log_param("system_prompt", system_prompt)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

# %%

query = 'मूल नियम ॥8 एवं मध्यप्रदेश सिवित्र सेवा (आचरण) नियम, 4965 के नियम 7 क्या कहता है विस्तार से बताओ?'

# if mlflow_lgging:
#    mlflow.log_param("query", query)

response = chain.invoke({"input": query})

# if mlflow_lgging:
#    mlflow.log_param("context", response['context'])
#    mlflow.log_param("response", response['answer'])

response

# end the logging of the mlflow
# mlflow.end_run()

# %%
'अनधिकृत अनुपस्थिति कौन से नियम में बात हो रही है?'
response

# %%
query = 'राज्य सचिव आदेश 4 क्या कहता है विस्तार से बताओ?'
response = chain.invoke({"input": query})
response

# %%
query = 'अनधिकृत अनुपस्थिति कौन से नियम में बात हो रही है?'
response = chain.invoke({"input": query})
response

# %%
query = 'राज्य सचिव आदेश 2 क्या कहता है मुझे संक्षेप में बताओ'
response = chain.invoke({"input": query})
response
