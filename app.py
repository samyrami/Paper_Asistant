# app.py
import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import re

# Load environment variables at startup
load_dotenv()

def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Error: OPENAI_API_KEY not found in environment variables")
        st.stop()
    return api_key

class PaperSection:
    def __init__(self, content="", page_number=None, references=None):
        self.content = content
        self.page_number = page_number
        self.references = references or []

class ResearchPaper:
    def __init__(self, filename):
        self.filename = filename
        self.title = PaperSection()
        self.abstract = PaperSection()
        self.introduction = PaperSection()
        self.methodology = PaperSection()
        self.results = PaperSection()
        self.discussion = PaperSection()
        self.findings = PaperSection()
        self.future_research = PaperSection()
        self.references = PaperSection()
        self.full_text = ""

class ResearchPaperProcessor:
    def __init__(self, pdf_directory="./Context_pdf"):
        self.pdf_directory = pdf_directory
        api_key = get_openai_api_key()
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=api_key,
            model="gpt-4"
        )

    def verify_response(self, response, source_docs):
        """Verify that the response matches the source documents."""
        verification_prompt = f"""
        Verify if this response is accurately supported by the provided source documents.
        If not, provide a corrected response using only information from the sources.
        
        Response: {response}
        
        Source documents: {[doc.page_content for doc in source_docs]}
        """
        
        verification = self.llm.invoke([
            SystemMessage(content="You are a fact-checker. Only include information explicitly supported by the source documents."),
            HumanMessage(content=verification_prompt)
        ])
        
        return verification.content

    def get_vector_store(self):
        """Creates or loads the vector store with enhanced metadata."""
        try:
            # Get list of currently selected documents from Streamlit session state
            if 'selected_papers' not in st.session_state:
                st.session_state.selected_papers = []
            
            documents = []
            
            # Only process selected documents
            for filename in st.session_state.selected_papers:
                if filename.endswith('.pdf'):
                    try:
                        file_path = os.path.join(self.pdf_directory, filename)
                        loader = PyPDFLoader(file_path)
                        pages = loader.load()
                        
                        for page in pages:
                            # Clean and verify metadata
                            cleaned_metadata = {
                                'source': filename,
                                'page': page.metadata.get('page', 1),
                                'total_pages': len(pages)
                            }
                            page.metadata = cleaned_metadata
                            documents.append(page)
                            
                    except Exception as e:
                        st.error(f"Error processing {filename}: {str(e)}")
                        continue

            if not documents:
                st.error("No content found in selected PDFs")
                return None

            # Create new vector store from selected documents
            texts = []
            metadatas = []
            
            for doc in documents:
                chunks = self.text_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    texts.append(chunk)
                    metadatas.append(doc.metadata)

            vector_store = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=metadatas
            )
            
            return vector_store
            
        except Exception as e:
            st.error(f"Error in vector store processing: {str(e)}")
            return None

    def process_paper(self, filename):
        """Processes a paper and extracts relevant information."""
        paper = self.load_paper(filename)
        if not paper:
            return None

        prompt = f"""
        Analyze the following academic paper and extract:
        1. Paper title
        2. Main objective
        3. Methodology used
        4. Identified gaps
        5. Key findings
        6. Future research directions

        Paper:
        {paper.full_text[:2000]}
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            analysis = response.content

            return {
                "filename": filename,
                "title": self.extract_from_analysis(analysis, "Title"),
                "objective": self.extract_from_analysis(analysis, "Objective"),
                "methodology": self.extract_from_analysis(analysis, "Methodology"),
                "gaps": self.extract_from_analysis(analysis, "Gaps"),
                "findings": self.extract_from_analysis(analysis, "Findings"),
                "future_research": self.extract_from_analysis(analysis, "Future research")
            }

        except Exception as e:
            st.error(f"Error processing paper {filename}: {str(e)}")
            return None

    def load_paper(self, filename):
        """Loads and processes a specific paper."""
        file_path = os.path.join(self.pdf_directory, filename)
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            paper = ResearchPaper(filename)
            paper.full_text = "\n".join(page.page_content for page in pages)
            
            return paper
            
        except Exception as e:
            st.error(f"Error loading paper {filename}: {str(e)}")
            return None

    def extract_from_analysis(self, analysis, section):
        """Extracts a specific section from the analysis."""
        pattern = f"{section}:?(.*?)(?=\n\n|$)"
        match = re.search(pattern, analysis, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else f"{section} not found"

def create_chat_interface(processor):
    st.subheader("Interactive Paper Chat")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your paper analysis assistant. What would you like to know about the selected documents?"
        }]
    
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )

    vector_store = processor.get_vector_store()
    if not vector_store:
        st.error("Could not load knowledge base")
        return

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=processor.llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=st.session_state.conversation_memory,
        return_source_documents=True,
        chain_type="stuff",
        verbose=True
    )

    messages_container = st.container()
    
    with messages_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    st.caption("Sources:")
                    for source in message["sources"]:
                        st.caption(f"📄 {source['document']} - Page {source['page']}")

    if prompt := st.chat_input("Write your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Searching answers..."):
                try:
                    result = qa_chain({"question": prompt})
                    initial_response = result['answer']
                    source_docs = result.get('source_documents', [])
                    
                    verified_response = processor.verify_response(initial_response, source_docs)
                    
                    sources = []
                    seen_sources = set()
                    
                    for doc in source_docs:
                        source_key = (doc.metadata['source'], doc.metadata['page'])
                        if source_key not in seen_sources:
                            seen_sources.add(source_key)
                            sources.append({
                                'document': doc.metadata['source'],
                                'page': doc.metadata['page']
                            })

                    message_placeholder.markdown(verified_response)
                    if sources:
                        st.caption("Sources:")
                        for source in sources:
                            st.caption(f"📄 {source['document']} - Page {source['page']}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": verified_response,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Sorry, there was an error processing your question. Please try rephrasing it."
                    })

        with st.expander("📝 Suggested Questions"):
            try:
                suggestions_prompt = f"""
                Based on the last response and conversation context, 
                suggest 3 questions that could be relevant to 
                deepen the analysis. The response was: {verified_response}
                """
                suggestions = processor.llm.invoke([
                    SystemMessage(content="You are an assistant that generates relevant and specific question suggestions."),
                    HumanMessage(content=suggestions_prompt)
                ])
                st.markdown(suggestions.content)
            except Exception as e:
                st.error("Could not generate question suggestions.")

        if st.button("🗑️ Clear History"):
            st.session_state.messages = []
            st.session_state.conversation_memory.clear()
            st.experimental_rerun()

def main():
    st.set_page_config(page_title="Research Paper Analyzer", layout="wide")
    st.title("Advanced Research Paper Analysis Assistant")
    
    api_key = get_openai_api_key()
    if not api_key:
        return
    
    try:
        processor = ResearchPaperProcessor()
        
        with st.sidebar:
            st.markdown("### Analysis Options")
            
            pdf_files = [f for f in os.listdir(processor.pdf_directory) if f.endswith('.pdf')]
            if not pdf_files:
                st.error("No PDF files found in the Context_pdf directory")
                return
                
            st.session_state.selected_papers = st.multiselect(
                "Select papers to analyze:",
                pdf_files
            )
            
            analysis_type = st.radio(
                "Analysis type:",
                ["Tabular Comparison", "Detailed Summary", "Gap Analysis", "Interactive Chat"]
            )
        
        if st.session_state.selected_papers:
            if analysis_type == "Interactive Chat":
                create_chat_interface(processor)
            else:
                with st.spinner("Processing papers..."):
                    results = []
                    for paper in st.session_state.selected_papers:
                        paper_info = processor.process_paper(paper)
                        if paper_info:
                            results.append(paper_info)
                    
                    if results:
                        if analysis_type == "Tabular Comparison":
                            df = pd.DataFrame(results)
                            st.dataframe(df)
                        elif analysis_type == "Detailed Summary":
                            for result in results:
                                st.subheader(f"Analysis of: {result['filename']}")
                                st.write("**Title:**", result['title'])
                                st.write("**Objective:**", result['objective'])
                                st.write("**Methodology:**", result['methodology'])
                                st.write("**Findings:**", result['findings'])
                                st.write("**Future Research:**", result['future_research'])
                                st.write("---")
                        else:  # Gap Analysis
                            st.subheader("Research Gap Analysis")
                            for result in results:
                                st.write(f"**Paper: {result['filename']}**")
                                st.write("Identified gaps:", result['gaps'])
                                st.write("Future research:", result['future_research'])
                                st.write("---")

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()