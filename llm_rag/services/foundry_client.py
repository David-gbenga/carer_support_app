

#import dependencies
import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from azure.identity.aio import ( DefaultAzureCredential,
    get_bearer_token_provider,)


from rag import RAGRetriever


#creation and implementation of the async foundry_client function
async def foundry_client() -> None:
    """
    Interactive CLI for the Carer Support RAG application.

    Flow:
        User question
            ↓
        rag.py retrieves Azure AI Search context
            ↓
        foundry_client.py sends context + question to the Responses API
            ↓
        grounded answer
    """
    # clear CLI screen 
    os.system("cls" if os.name =="nt" else "clear")
    
    credential = None
    asyn_client = None
    rag = None
    
    try:
        load_dotenv()
        
        
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        model_deployment = os.getenv("MODEL_DEPLOYMENT")
        embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT")
        azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        
        try:
            embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", '1536'))
            
        except ValueError as exc:
            raise ValueError( "EMBEDDING_DIMENSIONS must be an integer.") from exc
        
        required_values = {
            "AZURE_OPENAI_ENDPOINT": azure_openai_endpoint,
            "MODEL_DEPLOYMENT": model_deployment,
            "EMBEDDING_DEPLOYMENT": embedding_deployment,
            "AZURE_SEARCH_ENDPOINT": azure_search_endpoint,
            "AZURE_SEARCH_INDEX_NAME": azure_search_index_name,
        }
        
        
        #raise value error for variable names (endpoints/deployments) not configured
        for variable_name, value in required_values.items():
            if not value:
                raise ValueError(f"{variable_name} is not confirgured")
         
        
        #Created azure credential chain to authenticate via azure cli managed identity or other supported Extra ID sources
        credential =DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential,  "https://cognitiveservices.azure.com/.default",)
        
        async_client = AsyncOpenAI(base_url= azure_openai_endpoint, api_key= token_provider,)