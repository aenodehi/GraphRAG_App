import os
import json
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Neo4jVector
from langchain.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
import streamlit as st
import tempfile
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import requests

class CustomLLM:
    def __init__(self, api_url, model_name):
        self.api_url = api_url
        self.model_name = model_name

    def __call__(self, prompt, **kwargs):
        # Ensure prompt is a string
        if not isinstance(prompt, str):
            prompt = str(prompt)
        # Set max_length and max_new_tokens to avoid truncation issues
        kwargs.setdefault('max_length', 512)
        kwargs.setdefault('max_new_tokens', 256)

        # Prepare the payload for the Ollama API
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_length": kwargs['max_length'],
            "max_new_tokens": kwargs['max_new_tokens']
        }

        # Send the request to the Ollama API
        response = requests.post(self.api_url, json=payload, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            try:
                # Process the streaming response
                full_response = ""
                for chunk in response.iter_content(chunk_size=None):
                    if chunk:
                        full_response += chunk.decode('utf-8')
                # Split the full response by JSON objects
                json_objects = full_response.strip().split('\n')
                # Concatenate the JSON objects into a single valid JSON string
                concatenated_json = '[' + ','.join(json_objects) + ']'
                # Parse the concatenated JSON string
                parsed_response = json.loads(concatenated_json)
                # Extract the final response
                final_response = parsed_response[-1]['response']
                return final_response
            except json.JSONDecodeError as e:
                st.error(f"Failed to decode JSON: {e}")
                st.error(f"Raw response: {full_response}")
                raise Exception(f"Failed to decode JSON: {e}")
        else:
            raise Exception(f"Failed to generate text: {response.status_code} - {response.text}")

    def with_structured_output(self, *args, **kwargs):
        return self

def _convert_to_graph_document(raw_schema):
    # Print the raw_schema to inspect its value
    print(f"raw_schema: {raw_schema}")

    # Attempt to parse raw_schema if it's a string
    if isinstance(raw_schema, str):
        try:
            raw_schema = json.loads(raw_schema)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse raw_schema JSON")

    # Ensure raw_schema is a dictionary
    if not isinstance(raw_schema, dict):
        raise ValueError("raw_schema is not a dictionary")

    # Process raw_schema assuming it's now a dictionary
    if not raw_schema.get("parsed"):
        raise ValueError("Parsed key not found in raw_schema")

    # Proceed with the rest of the function
    # (Add your existing logic here to process the graph document)
    nodes = []
    relationships = []
    return nodes, relationships

def main():
    st.set_page_config(
        layout="wide",
        page_title="GraphRApp",
        page_icon=":graph:"
    )
    #st.sidebar.image('logo.png', use_column_width=True)
    with st.sidebar.expander("Expand Me"):
        st.markdown("""
    This application allows you to upload a PDF file, extract its content into a Neo4j graph database, and perform queries using natural language.
    It leverages LangChain and Ollama to generate Cypher queries that interact with the Neo4j database in real-time.
    """)
    st.title("GraphRApp: Realtime GraphRAG App")

    load_dotenv()

    # Set Ollama API URL and model name
    if 'OLLAMA_API_URL' not in st.session_state:
        st.sidebar.subheader("Ollama API URL")
        ollama_api_url = st.sidebar.text_input("Enter your Ollama API URL:", value="http://localhost:11434/api/generate")
        ollama_model_name = st.sidebar.text_input("Enter your Ollama Model Name:", value="llama3.1")
        if ollama_api_url and ollama_model_name:
            embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            llm = CustomLLM(ollama_api_url, ollama_model_name)  # Use Ollama API
            st.session_state['embeddings'] = embeddings
            st.session_state['llm'] = llm
            st.sidebar.success("Ollama API URL and Model Name set successfully.")
    else:
        embeddings = st.session_state['embeddings']
        llm = st.session_state['llm']

    # Initialize variables
    neo4j_url = None
    neo4j_username = None
    neo4j_password = None
    graph = None

    # Set Neo4j connection details
    if 'neo4j_connected' not in st.session_state:
        st.sidebar.subheader("Connect to Neo4j Database")
        neo4j_url = st.sidebar.text_input("Neo4j URL:", value="neo4j+s://<your-neo4j-url>")
        neo4j_username = st.sidebar.text_input("Neo4j Username:", value="neo4j")
        neo4j_password = st.sidebar.text_input("Neo4j Password:", type='password')
        connect_button = st.sidebar.button("Connect")
        if connect_button and neo4j_password:
            try:
                graph = Neo4jGraph(
                    url=neo4j_url,
                    username=neo4j_username,
                    password=neo4j_password
                )
                st.session_state['graph'] = graph
                st.session_state['neo4j_connected'] = True
                # Store connection parameters for later use
                st.session_state['neo4j_url'] = neo4j_url
                st.session_state['neo4j_username'] = neo4j_username
                st.session_state['neo4j_password'] = neo4j_password
                st.sidebar.success("Connected to Neo4j database.")
            except Exception as e:
                st.error(f"Failed to connect to Neo4j: {e}")
    else:
        graph = st.session_state['graph']
        neo4j_url = st.session_state['neo4j_url']
        neo4j_username = st.session_state['neo4j_username']
        neo4j_password = st.session_state['neo4j_password']

    # Ensure that the Neo4j connection is established before proceeding
    if graph is not None:
        # File uploader
        uploaded_file = st.file_uploader("Please select a PDF file.", type="pdf")

        if uploaded_file is not None and 'qa' not in st.session_state:
            with st.spinner("Processing the PDF..."):
                # Save uploaded file to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # Load and split the PDF
                loader = PyPDFLoader(tmp_file_path)
                pages = loader.load_and_split()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
                docs = text_splitter.split_documents(pages)

                lc_docs = []
                for doc in docs:
                    lc_docs.append(Document(page_content=doc.page_content.replace("\n", ""),
                    metadata={'source': uploaded_file.name}))

                # Print lc_docs to inspect its value
                print(f"lc_docs: {lc_docs}")

                # Clear the graph database
                cypher = """
                  MATCH (n)
                  DETACH DELETE n;
                """
                graph.query(cypher)

                # Define allowed nodes and relationships
                allowed_nodes = ["Patient", "Disease", "Medication", "Test", "Symptom", "Doctor"]
                allowed_relationships = ["HAS_DISEASE", "TAKES_MEDICATION", "UNDERWENT_TEST", "HAS_SYMPTOM", "TREATED_BY"]

                # Transform documents into graph documents
                transformer = LLMGraphTransformer(
                    llm=llm,
                    allowed_nodes=allowed_nodes,
                    allowed_relationships=allowed_relationships,
                    node_properties=False,
                    relationship_properties=False
                )

                graph_documents = transformer.convert_to_graph_documents(lc_docs)
                graph.add_graph_documents(graph_documents, include_source=True)

                # Use the stored connection parameters
                index = Neo4jVector.from_existing_graph(
                    embedding=embeddings,
                    url=neo4j_url,
                    username=neo4j_username,
                    password=neo4j_password,
                    database="neo4j",
                    node_label="Patient",  # Adjust node_label as needed
                    text_node_properties=["id", "text"],
                    embedding_node_property="embedding",
                    index_name="vector_index",
                    keyword_index_name="entity_index",
                    search_type="hybrid"
                )

                st.success(f"{uploaded_file.name} preparation is complete.")

                # Retrieve the graph schema
                schema = graph.get_schema

                # Set up the QA chain
                template = """
                Task: Generate a Cypher statement to query the graph database.

                Instructions:
                Use only relationship types and properties provided in schema.
                Do not use other relationship types or properties that are not provided.

                schema:
                {schema}

                Note: Do not include explanations or apologies in your answers.
                Do not answer questions that ask anything other than creating Cypher statements.
                Do not include any text other than generated Cypher statements.

                Question: {question}"""

                question_prompt = PromptTemplate(
                    template=template,
                    input_variables=["schema", "question"]
                )

                qa = GraphCypherQAChain.from_llm(
                    llm=llm,
                    graph=graph,
                    cypher_prompt=question_prompt,
                    verbose=True,
                    allow_dangerous_requests=True
                )
                st.session_state['qa'] = qa
    else:
        st.warning("Please connect to the Neo4j database before you can upload a PDF.")

    if 'qa' in st.session_state:
        st.subheader("Ask a Question")
        with st.form(key='question_form'):
            question = st.text_input("Enter your question:")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and question:
            with st.spinner("Generating answer..."):
                res = st.session_state['qa'].invoke({"query": question})
                st.write("\n**Answer:**\n" + res['result'])

if __name__ == "__main__":
    main()
