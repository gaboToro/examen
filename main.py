import streamlit as st  
import PyPDF2  
import chromadb  
from sentence_transformers import SentenceTransformer  
from dotenv import load_dotenv  
from groq import Groq  
import xml.etree.ElementTree as ET  
import pandas as pd  

load_dotenv()

client = chromadb.Client()  
qclient = Groq()  

st.title('PREDICCIÓN ELECTORAL 🇪🇨')

def delete_collection(client, collection_name):
    try:
        client.delete_collection(collection_name)
    except Exception as e:
        print(f"Error al eliminar la colección '{collection_name}': {e}")

def extract_pdf_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file) 
    return " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

def extract_text_from_xml(xml_file):
    tree = ET.parse(xml_file) 
    root = tree.getroot()
    return " ".join(elem.text.strip() for elem in root.iter() if elem.text)

def extract_text_from_xlsx(xlsx_file):
    df = pd.read_excel(xlsx_file)
    return df.to_string(index=False)

model = SentenceTransformer('all-mpnet-base-v2')

def get_embeddings(text):
    return model.encode([text])[0]

def save_text_in_chromadb(text):
    delete_collection(client, "documents")  
    collection = client.create_collection("documents")  
    embeddings = get_embeddings(text)  
    collection.add(documents=[text], embeddings=[embeddings], ids=["unique_document_id"])

def classify_vote(text):
    """Clasifica el texto en Voto Noboa, Voto Luisa o Voto Nulo basado en palabras clave."""
    keywords_noboa = ["Noboa", "Daniel", "alianza ADN", "voto Noboa"]
    keywords_luisa = ["Luisa", "González", "RC5", "voto Luisa"]
    keywords_nulo = ["nulo", "blanco", "voto nulo", "no válido"]

    text_lower = text.lower()
    
    if any(word.lower() in text_lower for word in keywords_noboa):
        return "Voto Noboa"
    elif any(word.lower() in text_lower for word in keywords_luisa):
        return "Voto Luisa"
    elif any(word.lower() in text_lower for word in keywords_nulo):
        return "Voto Nulo"
    else:
        return "No clasificado"

if 'messages' not in st.session_state:
    st.session_state.messages = []

to_display = st.session_state.messages.copy()
for message in to_display:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

uploaded_files = st.file_uploader('Sube archivos PDF/XML/XLSX', type=['pdf', 'xml', 'xlsx'], accept_multiple_files=True)

if uploaded_files:
    combined_text = ""
    for file in uploaded_files:
        if file.type == "application/pdf":
            combined_text += extract_pdf_text(file) + "\n"
        elif file.type == "application/xml":
            combined_text += extract_text_from_xml(file) + "\n"
        elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            combined_text += extract_text_from_xlsx(file) + "\n"

    st.session_state.document_content = combined_text
    save_text_in_chromadb(combined_text)

    # Clasificación de votos
    vote_category = classify_vote(combined_text)
    st.write(f"**Clasificación del voto:** {vote_category}")

if prompt := st.chat_input('¿En qué te puedo ayudar?'):
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('assistant'):
        if 'document_content' in st.session_state:
            combined_input = f"Document Content:\n{st.session_state.document_content}\n\nUser's Question: {prompt}"
            stream_response = qclient.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Responde basado en el documento."},
                    {"role": "user", "content": combined_input},
                ],
                model="llama-3.3-70b-versatile",
                stream=True
            )
            response = st.write_stream((chunk.choices[0].delta.content for chunk in stream_response if chunk.choices[0].delta.content))
        else:
            stream_response = qclient.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Responde normalmente."},
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.3-70b-versatile",
                stream=True
            )
            response = st.write_stream((chunk.choices[0].delta.content for chunk in stream_response if chunk.choices[0].delta.content))

        st.session_state.messages.append({'role': 'assistant', 'content': response})
