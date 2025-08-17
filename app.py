import streamlit as st
import os
from typing import Literal, List, Dict, Any
import time
import json
from datetime import datetime

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

st.set_page_config(
    page_title="🤖 Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
    }
    .main-header p {
        color: #f0f0f0;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .step-card {
        background: #5a3873;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-processing {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'processing_steps' not in st.session_state:
        st.session_state.processing_steps = []

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

class AgenticRAGSystem:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
    
        self.response_model = init_chat_model("gpt-4o-mini", temperature=0)
        self.grader_model = init_chat_model("gpt-4o-mini", temperature=0)
        
    
        self.vectorstore = None
        self.retriever_tool = None
        self.graph = None
        
    
        self.grade_prompt = (
            "You are a grader assessing relevance of a retrieved document to a user question. \n "
            "Here is the retrieved document: \n\n {context} \n\n"
            "Here is the user question: {question} \n"
            "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
            "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
        )
        
        self.rewrite_prompt = (
            "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
            "Here is the initial question:"
            "\n ------- \n"
            "{question}"
            "\n ------- \n"
            "Formulate an improved question:"
        )
        
        self.generate_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \n"
            "Context: {context}"
        )
    
    def process_documents(self, urls: List[str], progress_callback=None) -> Dict[str, Any]:
        """Process documents and create the RAG system"""
        try:
            steps = []
            
            if progress_callback:
                progress_callback(0.2, "Fetching documents from URLs...")
            
            docs = []
            for i, url in enumerate(urls):
                try:
                    doc = WebBaseLoader(url).load()
                    docs.extend(doc)
                    steps.append(f"✅ Successfully loaded document from {url}")
                except Exception as e:
                    steps.append(f"❌ Failed to load document from {url}: {str(e)}")
            
            if not docs:
                raise Exception("No documents were successfully loaded")
            
            if progress_callback:
                progress_callback(0.4, "Splitting documents into chunks...")
            
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=500, chunk_overlap=100
            )
            doc_splits = text_splitter.split_documents(docs)
            steps.append(f"✅ Split documents into {len(doc_splits)} chunks")
            

            if progress_callback:
                progress_callback(0.6, "Creating vector embeddings...")
            
            self.vectorstore = InMemoryVectorStore.from_documents(
                documents=doc_splits, embedding=OpenAIEmbeddings()
            )
            retriever = self.vectorstore.as_retriever()
            steps.append("✅ Created vector store with embeddings")
            
            if progress_callback:
                progress_callback(0.8, "Setting up retriever tool...")
            
            self.retriever_tool = create_retriever_tool(
                retriever,
                "retrieve_documents",
                "Search and return information from the processed documents.",
            )
            steps.append("✅ Created retriever tool")
            
            if progress_callback:
                progress_callback(0.9, "Building agentic workflow...")
            
            self._build_graph()
            steps.append("✅ Built agentic RAG workflow")
            
            if progress_callback:
                progress_callback(1.0, "System ready!")
            
            return {
                "success": True,
                "message": "RAG system initialized successfully",
                "steps": steps,
                "num_documents": len(docs),
                "num_chunks": len(doc_splits)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error initializing RAG system: {str(e)}",
                "steps": steps
            }
    
    def _build_graph(self):
        """Build the agentic RAG graph"""
        workflow = StateGraph(MessagesState)
        
        workflow.add_node("generate_query_or_respond", self.generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node("rewrite_question", self.rewrite_question)
        workflow.add_node("generate_answer", self.generate_answer)
        
        workflow.add_edge(START, "generate_query_or_respond")
        
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        
        workflow.add_conditional_edges(
            "retrieve",
            self.grade_documents,
        )
        
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        
        self.graph = workflow.compile()
    
    def generate_query_or_respond(self, state: MessagesState):
        """Generate query or respond directly"""
        response = (
            self.response_model
            .bind_tools([self.retriever_tool])
            .invoke(state["messages"])
        )
        return {"messages": [response]}
    
    def grade_documents(self, state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        """Grade retrieved documents for relevance"""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        
        prompt = self.grade_prompt.format(question=question, context=context)
        response = (
            self.grader_model
            .with_structured_output(GradeDocuments)
            .invoke([{"role": "user", "content": prompt}])
        )
        
        return "generate_answer" if response.binary_score == "yes" else "rewrite_question"
    
    def rewrite_question(self, state: MessagesState):
        """Rewrite the user question for better retrieval"""
        messages = state["messages"]
        question = messages[0].content
        prompt = self.rewrite_prompt.format(question=question)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [{"role": "user", "content": response.content}]}
    
    def generate_answer(self, state: MessagesState):
        """Generate final answer using retrieved context"""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = self.generate_prompt.format(question=question, context=context)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.graph:
            return {"success": False, "message": "RAG system not initialized"}
        
        try:
            messages = [{"role": "user", "content": question}]
            
            steps = []
            final_response = None
            
            for chunk in self.graph.stream({"messages": messages}):
                for node, update in chunk.items():
                    steps.append({
                        "node": node,
                        "message": update["messages"][-1],
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    if node in ["generate_query_or_respond", "generate_answer"]:
                        final_response = update["messages"][-1]
            
            return {
                "success": True,
                "response": final_response.content if final_response else "No response generated",
                "steps": steps
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error querying RAG system: {str(e)}"}

def main():
    init_session_state()
    
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Agentic RAG System</h1>
        <p>Intelligent Document Retrieval and Question Answering</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use the system"
        )
        
        st.divider()
        
        st.subheader("📚 Document Sources")
        
        default_urls = [
            "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
            "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
            "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
        ]
        
        urls = []
        for i in range(5):
            url = st.text_input(
                f"URL {i+1}",
                value=default_urls[i] if i < len(default_urls) else "",
                key=f"url_{i}"
            )
            if url.strip():
                urls.append(url.strip())
        
        st.divider()
        
        if st.button("🔄 Initialize RAG System", type="primary"):
            if not api_key:
                st.error("Please provide an OpenAI API key")
            elif not urls:
                st.error("Please provide at least one URL")
            else:
                initialize_rag_system(api_key, urls)
        
        st.subheader("📊 System Status")
        if st.session_state.rag_system:
            st.success("✅ System Ready")
        else:
            st.warning("⏳ System Not Initialized")
        
        if st.session_state.documents_processed:
            st.info(f"📄 Documents processed")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            
            with st.chat_message("assistant"):
                st.write(chat["response"])
                
                if chat.get("steps"):
                    with st.expander("View Processing Steps"):
                        for step in chat["steps"]:
                            st.text(f"[{step['timestamp']}] {step['node']}")
        
        if st.session_state.rag_system:
            question = st.chat_input("Ask a question about your documents...")
            
            if question:
                with st.chat_message("user"):
                    st.write(question)
                
                with st.chat_message("assistant"):
                    with st.spinner("Processing your question..."):
                        result = st.session_state.rag_system.query(question)
                    
                    if result["success"]:
                        st.write(result["response"])
                        
                        st.session_state.chat_history.append({
                            "question": question,
                            "response": result["response"],
                            "steps": result.get("steps", []),
                            "timestamp": datetime.now()
                        })
                        
                        if result.get("steps"):
                            with st.expander("View Processing Steps"):
                                for step in result["steps"]:
                                    st.text(f"[{step['timestamp']}] {step['node']}")
                        
                        st.rerun()
                    else:
                        st.error(result["message"])
        else:
            st.info("👆 Please initialize the RAG system first using the sidebar")
    
    with col2:
        st.header("🔄 Processing Log")
        
        if st.session_state.processing_steps:
            for step in st.session_state.processing_steps:
                st.markdown(f'<div class="step-card">{step}</div>', unsafe_allow_html=True)
        else:
            st.info("No processing steps yet")
        
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

def initialize_rag_system(api_key: str, urls: List[str]):
    """Initialize the RAG system with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(progress: float, message: str):
        progress_bar.progress(progress)
        status_text.text(message)
    
    rag_system = AgenticRAGSystem(api_key)
    
    result = rag_system.process_documents(urls, update_progress)
    
    if result["success"]:
        st.session_state.rag_system = rag_system
        st.session_state.documents_processed = True
        st.session_state.processing_steps = result["steps"]
        st.success(f"✅ {result['message']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", result["num_documents"])
        with col2:
            st.metric("Chunks", result["num_chunks"])
        with col3:
            st.metric("URLs", len(urls))
        
    else:
        st.error(f"❌ {result['message']}")
        if result.get("steps"):
            st.session_state.processing_steps = result["steps"]
    
    progress_bar.empty()
    status_text.empty()

if __name__ == "__main__":

    main()
