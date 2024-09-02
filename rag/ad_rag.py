# https://docs.llamaindex.ai/en/stable/examples/query_transformations/query_transform_cookbook/
# https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/


import chromadb
import pandas as pd
# LLama 
# ------------------------------------------------------------------------------

# LLama index
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    PromptHelper,
    ServiceContext,
    set_global_service_context
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline

from llama_index.core.tools import QueryEngineTool,ToolMetadata
from llama_index.core.query_engine import (
    SubQuestionQueryEngine,
    TransformQueryEngine,
    RetrieverQueryEngine,
)
from llama_index.core.indices.query.query_transform import (
    HyDEQueryTransform,
)
from llama_index.core.retrievers import QueryFusionRetriever



# local packages
# -----------------------------------------------------------
#from .normalize_text import normalize
#from .utils import load_config



class AdvancedRAG():
    def __init__(
        self,
        embedding_model=None,
        persist_directory="",
        query_method="native",
        similarity_top_k=10,
        llm=None,
    ):
        
        """
        The AdvancedRAG class is responsible for creating the baseline RAG ways for the multi-hop rag test
        
        Tips: You should provided the OPENAI_API_KEY to query functions 

        Args:
            embedding_model (_type_): The embedding model to be used to generate the Vector Database 
            file_path (str, optional): The file path to load. Defaults to "".
            persist_directory (str, optional): The persist directory of the vector database. Defaults to "".
            query_method (str, optional): The query method used to test. Defaults to "native"
                query_method should be one of ['native','hyde','sub_question',"multi_query"]
            similarity_top_k (int, optional): The similarity top k for the retrieval documents.
            llm (str, optional): The llm used for rewriting the query. Defaults to
                
        if the persist directory is None, which means you need to create the vector database
            
        """
        
        # Basic configuration
        # ---------------------------------------
        self.embedding_model=embedding_model
        self.similarity_top_k=similarity_top_k
    
        
        if llm:
            self.llm = llm
        
        # Query method in the vector database
        if query_method not in ['native','hyde','sub_question',"multi_query"]:
            raise ValueError("query_method must be one of 'native','hyde','sub_question','multi_query'")
        
        self.query_method = query_method
        
        
        if persist_directory:
            
            chroma_client = chromadb.PersistentClient(path=persist_directory)
            chroma_collection = chroma_client.get_or_create_collection("classical")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embedding_model
            )
            
            self.__set_query_engine()

    def from_documents(
        self,
        documents,
        persist_directory="",
        chunk_size=256
    ):
        """Load the documents to create a classic Vector Database

        Args:
            file_path (str, optional): _description_. Defaults to "".
        
        """
        
        # Initialize a text splitter with hardcoded values for chunking documents
        text_splitter = SentenceSplitter(chunk_size=chunk_size)
        prompt_helper = PromptHelper(
            context_window=2048,
            num_output=256,
            chunk_overlap_ratio=0.1,
            chunk_size_limit=None,
        )
        service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embedding_model,
            text_splitter=text_splitter,
            prompt_helper=prompt_helper,
        )
        set_global_service_context(service_context)
        transformations = [text_splitter] 
        pipeline = IngestionPipeline(transformations=transformations)
        nodes = pipeline.run(documents=documents)
        
        # Chromadb
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        chroma_collection = chroma_client.get_or_create_collection("classical")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        self.index = VectorStoreIndex(
            nodes, 
            storage_context=storage_context,
            embed_model=self.embedding_model,
            use_async=True,
            show_progress=True
        )
        
        self.__set_query_engine()
        
    
    # Set query
    # ---------------------------------------
    def __set_query_engine(
        self
    ):
        
        query_engine = self.index.as_query_engine(
                llm = self.llm,
                similarity_top_k=self.similarity_top_k,
                embed_model=self.embedding_model
            )
        
        #if query_method not in ['native','hyde','sub_question','query_rewrite']:
        if self.query_method == "native":
            self.query_engine = query_engine
        elif self.query_method == "hyde":
            self.query_engine = self.__set_query_engine_hyde(query_engine=query_engine)
        elif self.query_method == "sub_question":
            self.query_engine = self.__set_query_engine_sub(query_engine=query_engine)
        elif self.query_method == "multi_query":
            self.query_engine = self.__set_query_engine_multi_query()
        
    # HyDE
    def __set_query_engine_hyde(
        self,
        query_engine = None
    ):
        
        hyde = HyDEQueryTransform(include_original=True)
        query_engine_hyde = TransformQueryEngine(query_engine, hyde)
        
        return query_engine_hyde
    
    # Sub question
    def __set_query_engine_sub(
        self,
        query_engine = None
    ):
        query_engine_tools = [
            QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name="pg_essay",
                    description="Paul Graham essay on What I Worked On",
                ),
            ),
        ]
        
        query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools, 
            use_async=True,
            verbose=False,
        )
        
        return query_engine
        
    # multi query
    def __set_query_engine_multi_query(
        self,
    ):
        retriever = QueryFusionRetriever(
            [self.index.as_retriever()],
            similarity_top_k=self.similarity_top_k,
            num_queries=4,  # set this to 1 to disable query generation
            use_async=False,
            verbose=False
            # query_gen_prompt="...",  # we could override the query generation prompt here
        )
        
        query_engine_multi_query = RetrieverQueryEngine.from_args(retriever,
                                                          verbose=True,
                                                          embed_model=self.embedding_model,
                                                          llm=self.llm)
        
        return query_engine_multi_query

        
        
    # Query function
    # ------------------------------------
    def query(
        self,
        input="",
    ):
        """The rage query function

        Args:
            input (str, optional): input query. Defaults to "".

        Returns:
            response : the response from the query_engineer
        """
        
        response = self.query_engine.query(input)

        print(response)
        return response.source_nodes
    
    
    

"""
if __name__ == "__main__":
    
    from llama_index.llms.openai import OpenAI
    
    from utils import load_config
    
    load_config()
    
    from utils import load_hf_model
    llama_embedding_model = load_hf_model(
        model_name="facebook/contriever-msmarco",
        repo="../repo",
    )
    
    # ChatGPT for the advanced rag for document summarization
    chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo")
    
    classical_native_rag = AdvancedRAG(
        embedding_model=llama_embedding_model,
        persist_directory="../db/multi_hop",
        query_method="native",
        llm=chatgpt,
        similarity_top_k=10
    )
    
    from utils import Loader
    
    multi_hop_loader = Loader(
        is_wiki=False
    )

    
    # multi query 要重新设立
    classical_native_rag.from_documents(
        documents=multi_hop_loader.load_data("../data/multi_hop_rag/corpus.json"),
        persist_directory="../db/multi_hop",
        chunk_size=256
    )
    
    
    import json
    with open('../data/multi_hop_rag/MultiHopRAG.json', 'r') as file:
        query_data = json.load(file)
        
    from evaluate_rag import run_rag_experiment
    
    run_rag_experiment(
        experiment_name="test",
        query_data=query_data,
        verbose=True,
        results_dir="./test",
        is_ad_rag=True,
        rag=classical_native_rag
    )
    
    
    
    with open('./test/test_retrieval.json', 'r') as file:
        retrieval_save_list = json.load(file)
        
    
    retrieved_lists = []
    gold_lists  = []
    for d in retrieval_save_list:
        if d['question_type'] == 'null_query':
            continue
        retrieved_lists.append([m['text'] for m in d['retrieval_list']])
        gold_lists.append([m['fact'] for m in d['gold_list']])
    
    from evaluate_rag import calculate_metrics
    print(calculate_metrics(retrieved_lists,gold_lists))
"""   
    
    
    
    

