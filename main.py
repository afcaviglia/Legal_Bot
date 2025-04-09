
#####################################################################################
##################################### *IMPORTS* #####################################
#####################################################################################

from fastapi import FastAPI, Request    ##Install from requierments
#from fastapi import HTTPException, Query
import json
from uuid import uuid4                  ##Install from requierments
from pydantic import BaseModel
import cohere                           ##Install from requierments
from sklearn.metrics.pairwise import cosine_similarity  ##Install from requierments
from dotenv import load_dotenv          ##Install from requierments
import os
import pypdf
#web scrapping
import requests
from bs4 import BeautifulSoup
import re
import requests
import requests
from bs4 import BeautifulSoup
from time import sleep
from langchain_text_splitters import RecursiveCharacterTextSplitter;

#######################################################################################
##################################### *FUNCTIONS* #####################################
#######################################################################################

############################
######## PDF2chunks ########
############################

### Carga de documentos
# Creamos un objeto de lectura
def load_pdf(path_pdf):
    """Str ---> Str
    Función para cargar el contenido de un archivo PDF
    Abre el PDF del path provisto en el input y extrae el texto de todas las páginas"""
    pdf = pypdf.PdfReader(path_pdf)
    historias = "".join([pdf.pages[pag].extract_text() for pag in range(len(pdf.pages))])
    return historias

def histories_splitter(historias, chunk_size):
    """Str -> Int ---> [Str]
    Divide el texto provisto en fragmentos de texto más pequeños ("chunks"). La longtud de /
    estos fragmentos se especifica con chunk_size (Int).
    Devuelve una lista de Strings.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore

    text_splitter = RecursiveCharacterTextSplitter(
    #Creamos una instancia del splitter en la cual especificamos el tamaño objetivo de los chunks,/
    # su superposición y el método para determinar el largo del chunk (en este caso len, la cantidad de caracteres)
        chunk_size = chunk_size,
        chunk_overlap  = int(chunk_size * 0.1), # Aplicamos una superposición del 10% del tamaño del chunk por convención
        length_function = len,
    )

    return text_splitter.split_text(historias)

def splitt(content: str):
    """Str ---> [Str]
    Procesa el contenido en chunks de unos 2300 caraceres,/
    lo cual equivale aproximadamente a 530 tokens, con un /
    overlap (sobreposición) de los chunks del 10%.
    """
    chunk_size=2300
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size//10)
    texts = text_splitter.split_text(content)
    return texts


#################################
######### Base de Datos #########
#################################

def save_doc(input_doc, save_local=True):
    """[Str] ---> [Str]
    Almacena los textos de la lista dada como argumento en un diccionario /
    de objetos json. Este diccionario es "documents" y por defecto, guarda /
    los archivos json en la carpeta ./resources
    """
    #id = len(docuemnts_ids)
    chunks = splitt(input_doc)
    ids = []
    for chunk in chunks:
        doc_id = str(uuid4()) #Generamos un id único para cada chunk
        document = {}
        document["doc_id"] = f"{doc_id}"
        #document["title"] = input_doc.title    
        document["content"] = chunk
        documents[doc_id] = document #Agregamos el dicionario generado para el chunk al diccionario de documentos
        #Si el argumento save_local == True (valor por defeto), se guardan localmente los diccionarios de los chunks
        if save_local:
            with open(f"./resources/{doc_id}.json", "w") as fp:
                json.dump(document, fp)
            print(f"Document {doc_id} locally saved.")


        ids.append(doc_id) #para devolver al cliente cuáles fueron los docs guardados, y si fueron 1 o más
    if len(ids) == 1:
        return ids, {"message": "Document successfully uploaded",
                "document_id": f"{ids[0]}"}
    elif len(ids)>1:
        return ids, {"message": "Document too long: It was splitted and successfully uploaded",
                "documents_ids": f"{ids}"}


def save_new_doc(input_doc, save_local=True):
    """[Str] ---> [Str]
    Almacena los textos de la lista dada como argumento en un diccionario /
    de objetos json. Este diccionario es "new_documents" y por defecto, guarda /
    los archivos json en la carpeta ./new_resources
    A diferencia de "save_doc" se encarga de administrar la información adicional /
    generada por el módulo de augmentación.
    """
    #id = len(docuemnts_ids)
    chunks = splitt(input_doc)
    ids = []
    for chunk in chunks:
        doc_id = str(uuid4()) #Generamos un id único para cada chunk
        document = {}
        document["doc_id"] = f"{doc_id}"
        document["content"] = chunk
        new_documents[doc_id] = document #Agregamos el dicionario generado para el chunk al diccionario de documentos
        #Si el argumento save_local == True (valor por defeto), se guardan localmente los diccionarios de los chunks
        if save_local:
            with open(f"./new_resources/{doc_id}.json", "w") as fp:
                json.dump(document, fp)
            print(f"Document {doc_id} locally saved.")

        ids.append(doc_id) #para devolver al cliente cuáles fueron los docs guardados, y si fueron 1 o más
    if len(ids) == 1:
        return ids, {"message": "Document successfully uploaded",
                "document_id": f"{ids[0]}"}
    elif len(ids)>1:
        return ids, {"message": "Document too long: It was splitted and successfully uploaded",
                "documents_ids": f"{ids}"}


##################
### Embeddings ###
##################

def embedd_doc(doc_id: str, save_local=True):
    """Genera los embeddings de los objetos json almacenados en la base de datos /
    de jurisprudencia en el diccionario "documents" y los almacena en el /
    diccionario "embeddings". Si el argumento save_local /
    tiene valor True, los guarda en formato json en la carpeta ./embeddings
    """
    embeddings[doc_id] = co.embed(
            texts=[documents[doc_id]["content"]],
            model="embed-multilingual-v3.0",
            input_type="search_document",
            embedding_types=["float"]).embeddings.float_[0]
    if save_local:
        with open(f"./embeddings/{doc_id}.json", "w") as fp:
            json.dump(embeddings[doc_id], fp)
        print(f"Embedding {doc_id} locally saved.")
    return {"Message": f"Embeddings generated successufully for document {doc_id}"}

def embedd_new_doc(doc_id: str, save_local=True):
    """Genera los embeddings de los objetos json almacenados en la base de datos /
    de información legal ampliada por el módulo de augmentación en el diccionario /
    "new_documents" y los almacena en el diccionario "new_embeddings". Si el argumento save_local /
    tiene valor True, los guarda en formato json en la carpeta ./embeddings
    """
    new_embeddings[doc_id] = co.embed(
            texts=[new_documents[doc_id]["content"]],
            model="embed-multilingual-v3.0",
            input_type="search_document",
            embedding_types=["float"]).embeddings.float_[0]
    if save_local: #almacenamiento local de los embeddings
        with open(f"./new_embeddings/{doc_id}.json", "w") as fp:
            json.dump(new_embeddings[doc_id], fp)
        print(f"Embedding {doc_id} locally saved.")
    return {"Message": f"Embeddings generated successufully for document {doc_id}"}

def embedd_query(query: str):
    """ Str ---> [float]
    Genera el embedding de un string dado por argumento.
    """
    query_embedding = co.embed(
                        texts=[query],
                        model="embed-multilingual-v3.0",
                        input_type="search_query", 
                        embedding_types=["float"]).embeddings.float_[0],
    return query_embedding

###########################
### Carga de json_files ###
###########################

def load_json_files(directory):
    """ Str ---> {Set}
    Carga todos los archivos json almacenados en el directorio /
    dado por argumento.
    Retorna todo el conjunto de datos como un diccionario de objetos json.
    """
    docs = {}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_name=filename[:-5]
            with open(os.path.join(directory, filename), 'r') as fp:
                docs[file_name]=json.load(fp)
    return docs

def remove_duplicates(documents):
    """{Set} ---> {Set}
    Dado un diccionario de objetos por argumento, devuelve el diccionario sin /
    repetición del contenido.
    Importante para hacer una limpieza de datos en caso de haber generado más de /
    un embedding para un único texto original.
    """
    unique_contents = {}
    unique_documents = {}
    for doc in documents:  #Comparación exhaustiva. De fuerza bruta
        content = doc.get("content")
        if content not in unique_contents:
            unique_contents.add(content)
            unique_documents.add(doc) 
    return unique_documents

##############################################
### Busqueda de información para responder ###
##############################################

def search_documents(query_embedding, embeds, docs, top_n=5):
    """[Float] -> {Set} -> {Set} ---> {Set}
    Dado un vector (por ejemplo de un prompt), devuelve los documentos más similares /
    almacenados en la base de datos. Para esto, toma también la base de datos vectorial /
    "embeddings" o "new_embeddings" como argumento. La cantidad de documentos que la función /
    retorna se puede establecer por "top_n" (por defecto 5).
    """
    similaridades = []  # Lista para almacenar los resultados de similitud
    docs_ids = []  # Para almacenar los ids de los chunks que se van analizando

    for doc_id in embeds:
        # Calculamos la distancia entre la query y los embeddings
        sim_coseno = cosine_similarity(query_embedding, [embeds[doc_id]])
        
        # Acceder al valor de la similitud (un solo valor en la matriz 1x1)
        similaridades.append(sim_coseno[0][0])
        docs_ids.append(doc_id)

    # Encontrar los documentos más similares
    sims_and_ids = [(sim, doc_id) for sim, doc_id in zip(similaridades, docs_ids)]
    sims_and_ids.sort(reverse=True)  # Ordenar la lista de duplas (similitud, id_doc) de mayor a menor similitud

    # Seleccionar los N documentos más similares
    most_similar_documents = [docs[doc_id] for sim, doc_id in sims_and_ids[:top_n]]

    return most_similar_documents


######################
### Web scrapping  ###
######################

from  enum import Enum

class jurisdiccion(Enum):
    nacional = "nacional"
    provincial = "provincial"
    
class tipo_norma(Enum):
    ley = "leyes"
    decreto = "decretos"
    decisiones_administrativas = "decisiones_administrativas"
    resolucion = "resoluciones"
    disposicion = "disposiciones"
    acta = "actas"
    actuacion = "actuaciones"
    acuerdo = "acuerdos"
    circular = "circulares"
    comunicacion = "comunicaciones"
    comunicado =  "comunicados"
    convenio = "convenios"
    #por ahora hasta acá.No creo que hagan falta el resto de tipos
    
def buscar_normativas(jurisdiccion, numero_norma, tipo_norma=None, limit=50, offset=1):
    """jurisdiccion -> Int -> tipo_norma ---> Str

    Busca una norma en la página web oficial de la República Argentina.
    Parámetros:
        jurisdiccion (jurisdiccion): jurisdicción nacional o provincial de la norma.
        numero_norma (int): Número específico de la norma (ley, decreto, etc).
        tipo_norma (tipo_norma): si la norma que se busca es una ley, decreto, comunicación, etc.
    Retorna:
        str: link de la norma buscada o un mensaje si no se la ha encontrado.
    """
    base_url = "https://www.argentina.gob.ar/normativa/buscar"
    
    # Construir los parámetros de la URL
    params = {
        "jurisdiccion": jurisdiccion,
        "numero": numero_norma,
        "limit": limit,
        "offset": offset
    }
    
    if tipo_norma:
        params["tipo_norma"] = tipo_norma

    # Realizar la solicitud GET
    response = requests.get(base_url, params=params)
      
    # Verificar el código de estado de la respuesta
    if response.status_code == 200:
        try: 
            patron = "norma encontrada en"
            resultado=re.split(patron, response.text, 1)[1] #Descartamos todo lo anterior a la lista de normas
            resultado_url = extraer_primer_resultado(resultado) # Obtenemos específicamente el url de la ley
            return resultado_url
        except IndexError:
            """Si no encuentra el patrón de búsqueda de resultados, devuelve un IndexError y significa que la norma no fue encontrada"""
            return  f"Error: la norma {params["jurisdiccion"]} {params["tipo_norma"]} {params["numero"]} no fue enonctrada"
    else:
        """ """
        return f"Error: {response.status_code}, {response.text}"

def extraer_primer_resultado(html):
    """ Str ---> Str
    Luego de realizar una búsqueda de normas, dado el link del /
    resultado de dicha búsqueda como argumento, devuelve el link /
    de acceso a la primera norma en la lista de resultados.
    """
    soup = BeautifulSoup(html, "html.parser")
    normas_section = soup.find(id="normas")
    
    if normas_section is None:
        return "No se encontró la sección de normas."

    # Buscar el primer enlace en los resultados dentro del tbody
    first_result = normas_section.find("tbody").find("a", href=True)
    if first_result:
        link = first_result['href']
        return f"https://www.argentina.gob.ar{link}"
    else:
        return "No se encontró el primer resultado."

def extraer_texto_norma(url):
    """Str ---> Str
    Dada la página web con el texto de una norma, /
    extrae su contenido en un string.
    """
    # Hacer la solicitud a la página web
    try:
        url=url+"/texto" #la terminación "/texto" es requerida para ingresar al contenido de la ley
        response = requests.get(url)
    except:
        return "Imposible procesar una ley que no fue encontrada"
    # Verificar que la solicitud fue exitosa
    if response.status_code == 200:
        # Parsear el contenido HTML de la página
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extraer todo el texto de la página
        texto = soup.get_text(separator=' ', strip=True)
        return texto
    else:
        return f"Error: {response.status_code}"

def procesar_nueva_norma(texto: str):
    if texto == "Imposible procesar una ley que no fue encontrada":
        return
    new_ids, _ = save_new_doc(texto)
    for doc_id in new_ids:
        embedd_new_doc(doc_id)
        sleep(0.2)
    return {"message": "norma procesada con exito"}


#####################################
########### Tools Calling ###########
#####################################

# Diccionario de funciones
funciones = {
    "buscar_normativas": buscar_normativas
}
# descripciones de herramientas a las que el modelo tiene acceso:
tools = [
    {
        "type": "function",
        "function": {
            "name": "buscar_normativas",
            "description": "Busca una norma en la página web oficial de la República Argentina.",
            "parameters": {
                "type": "object",
                "properties": {
                    "jurisdiccion": {
                        "type": "string",
                        "description": "Elige la jurisdicción de la norma ('nacional', 'provincial')"
                    },
                    "numero_norma": {
                        "type": "integer",
                        "description": "Número de la norma que se busca."
                    },
                    "tipo_norma": {
                        "type": "string",
                        "description": "Tipo de la norma ('leyes', 'decretos', 'decisiones_administrativas', 'resoluciones', 'disposiciones', 'actas', 'actuaciones', 'acuerdos', 'circulares', 'comunicaciones', 'comunicados', 'convenios')."
                    }
                },
                "required": ["jurisdiccion", "numero_norma"]
            }
        }
    }]


######################################
############ LLMs calling ############
######################################

# Función para formatear el prompt de respuesta
#---------------------------------- Para el llamado al primer LLM: jurisprudencia y dogma ----------------------------------#
def prelude_jurisprudencia():
    return """
        ### Eres un abogado. Tu propósito es asesorar y responder legalmente a las consultas de los usuarios. ###

        ### Las respuestas tienen que tener las siguientes caracteristicas:
        - Hacer referencia a las normas y jurisprudencias correspondientes, e incluir citas textuales./
        - Dar recomendaciones específicas en base a normas y jurisprudencias./
        - Si la consulta del usuario no se refiere a ningun accidente de tránsito, responder "Soy un chat bot legal, si tienes alguna consulta hazmelo saber. Saludos!". /
        - Aconsejar por asesoramiento legal con un abogado o abogada.

        ### Guía de estilo:
        - Responder de manera seria y concisa, como si la persona tuviera urgencia.
        - Siempre debes responder en español.
        """
def prompt_jurisprudencia(docs, query):
    return f"""
    # Si es posible, responde la pregunta del usuario en base la siguiente inormación de jurisprudencia o dogma judicial:
    # Fragmento:
    < {docs} >
    # Pregunta del usuario:
    [{query}]
    """
#---------------------------------- Para el llamado al segundo LLM: web scraping de normas ----------------------------------#
def prelude_web_scraping():
    return f"""
        ### Tu propósito es buscar el link a las leyes que se mencionan en el texto. ###
        
        ### Para eso, debes llamar a la función provista buscar_normativas, con los parámetros de los que se disponga en el texto:
        - jurisdicción (nacional o provincial)/
        - tipo de normativa/
        - Número de norma./
    """

#---------------------------------- Para el llamado al tercer LLM: corrección y ampliación ----------------------------------#
def prompt_advisory(scrape, response_jurisprudencia):
    return f"""
    #### Mejora y corrige el siguiente texto de asesoría legal en base a las Normas provistas:

    ### Normas
    [{scrape}]
    
    ### Texto legal para mejorar:
    {response_jurisprudencia}

    ### Texto legal mejorado y corregido:
    
    """
#---------------------------------- Llamado al primer LLM ----------------------------------#
def llm_jurisprudencia(historial_de_chat, query, embeddings, documents, modelo="command-r-plus-08-2024"):
    """{Set} -> Str -> {Set} -> {Set} ---> Str -> {Set}
    Realiza una consulta al LLM. Se le provée el historial completo de la conversación /
    y la consulta del "user", junto con las bases de datos de embeddings y textos correspondientes.
    Devuelve la respuesta del modelo, junto con el historial actualizado
    """
    relevant_docs = search_documents(embedd_query(query), embeddings, documents, top_n=5)
    context = " ".join([doc["content"] for doc in relevant_docs]) # unir fragmentos: para largos de chunk pequeños
    prompt = prompt_jurisprudencia(context, query)
    
    # Ordenes de system más el prompt para este modelo específicamente (sin afectar a los siguientes)
    Sys_and_Prompt = [{"role": "system", "content": prelude_jurisprudencia()},
                  {"role": "user", "content": prompt}]
    
    response = co.chat(
        model=modelo,
        messages = historial_de_chat + Sys_and_Prompt,
        seed = 16
    )
    historial_de_chat.append({"role": "user", "content": query})
    
    print(f"HISTORIAL DE CHAT: {historial_de_chat}")
    print(f"RESPUESTA: {response.message.content[0].text}")
    return response.message.content[0].text, historial_de_chat

#---------------------------------- Llamado al segundo LLM ----------------------------------#
def llm_web_scraper(prompt_ws, developer=True):
    """Str ---> {Set}
    Se le provee al LLM la respuesta generada en base a la jursiprudencia por llm_jurisprudencia() /
    En base a este texto realiza una búsqueda de leyes en el buscador oficial de leyes argentinas /
    Llama a las funciones responsables de extraer la información de la norma y almacena la información /
    en las bases de datos de documentos adicionales (new_documents) y de sus embeddings correspondientes /
    (new_embeddings)
    """
    #print("#-"*50)
    #print(prompt_ws)
    #print("#-"*50)
    if developer:
        new_embeddings= load_json_files(directory_new_embs)
        new_documents= load_json_files(directory_new_docs)
        
        relevant_new_docs = search_documents(embedd_query(prompt_ws), new_embeddings, new_documents, top_n=5)
        tool_content = []
        tool_content.append(json.dumps(relevant_new_docs))
        
    else:
        prelude_ws = prelude_web_scraping()
    
        messages=[{"role": "system", "content": prelude_ws},
                  {"role": "user", "content": prompt_ws}]
        
        response = co.chat(
            model="command-r-plus-08-2024",  # Preferible para respuestas en español
            messages=messages,
            tools=tools
        )
    
        # Depuración: Imprimir las herramientas que el modelo está llamando
        print(f"Tool calls: {response.message.tool_calls}")
    
        messages.append({'role': 'assistant', 'tool_calls': response.message.tool_calls, 'tool_plan': response.message.tool_plan})
    
        tool_content = []
        # Iteración sobre las llamadas de herramientas generadas por el modelo
        if response.message.tool_calls: #Si no llama a ninguna función pasa a devolver una lista vacía...
            for tc in response.message.tool_calls:
                print(f"Calling function: {tc.function.name} with arguments: {tc.function.arguments}")
                
                # Llamada a la herramienta recomendada por el modelo
                tool_result = funciones[tc.function.name](**json.loads(tc.function.arguments))
        
                # Extracción del texto de la norma
                texto_norma = extraer_texto_norma(tool_result)
        
                # Procesar la norma
                procesar_nueva_norma(texto_norma)
        
                # Aquí aseguramos que new_embeddings y new_documents se inicialicen 
                new_embeddings = load_json_files(directory_new_embs)
                new_documents = load_json_files(directory_new_docs)

                relevant_new_docs = search_documents(embedd_query(prompt_ws), new_embeddings, new_documents, top_n=5)
        
                # Guardado del output en una lista
                tool_content.append(json.dumps(relevant_new_docs))
                #messages.append({'role': 'tool', 'tool_call_id': tc.id, 'content': tool_content})
        
    return tool_content

#---------------------------------- Llamado al tercer LLM ----------------------------------#
def llm_advisory(historial_de_chat, response_jurisprudencia, scrape, haiku=False):
    """{Set} -> Str -> {Set} ---> Str -> {Set}
    Realiza una curaduría de la respuesta incial generada por la jurisprudencia /
    (argumento "response-jurisprudencia") a la luz /
    de la información legal ampliada y actualizada por el web scraping (argumento "scrape").
    Devuelve la respuesta curada y el historial de chat actualizado.
    """
    if haiku:
        prelude_advisory = """
        ### Eres un abogado. Tu propósito es mejorar la asesoría legal provisa en la consulta del usuario. ###

        ### Las respuestas tienen que tener las siguientes caracteristicas:
        - Ofrecer información breve e incluir citas textuales a las normas./
        - Agregar recomendaciones específicas en base a leyes y jurisprudencias./
        - Si el texto no tiene ninguna referencia útil a leyes, resumir  el  mensaje./
        - Al final, preguntar por detalles sobre la situación que se está asesorando.
        
        ### Guía de estilo:
        - Siempre debes responder de forma concisa y en español.
        """
    else:
        prelude_advisory = """
        ### Eres un abogado especializado en corregir asesorías legales. Tu propósito es mejorar y corregir textos de/
        ###  asesoría legal en base a las normativas provistas en la consulta. Tu tarea es:
        - Agregar artículos relevantes de las normas mencionadas, e incluir citas textuales./
        - Agregar recomendaciones específicas en base a leyes y jurisprudencias./
        - Si el texto no tiene ninguna referencia relativa a las leyes, devuelve el texto igual./
        - Si es necesario, al final, preguntar por detalles sobre la situación que se está asesorando.
        
        ### Guía de estilo:
        - Siempre debes ampliar la información en español.
        """
    
    Sys_and_Prompt= [{"role": "system", "content": prelude_advisory},
                    {"role": "user", "content": prompt_advisory(scrape, response_jurisprudencia)}]
    
    response = co.chat(
        model="command-r-plus-08-2024",
        messages=Sys_and_Prompt,
        seed = 14
    )

    historial_de_chat.append({"role": "assistant", "content": response.message.content[0].text})
    
    return response.message.content[0].text, historial_de_chat




def generate_answer(prompt: str, developer=False, haiku=False):
    """Str -> Str
    Función main del código. Integra el ciclo completo de la arquitectura RAG.
    """
    global historial_de_chat
    res_jurisprudencia, historial_de_chat = llm_jurisprudencia(historial_de_chat,
                                                                prompt.prompt,
                                                                embeddings, 
                                                                documents, 
                                                                modelo="command-r-plus-08-2024"
                                                                )
    
    normas_adicionales = llm_web_scraper(prompt_ws = res_jurisprudencia,
                                          developer=developer)
    
    answer, historial_de_chat = llm_advisory(historial_de_chat, response_jurisprudencia = res_jurisprudencia, 
                 scrape = normas_adicionales, 
                 haiku=haiku)
    
    return answer


########################################################################################
######################################## *INIT* ########################################
########################################################################################

# Inicialización de FastAPI
app = FastAPI()

#load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2()

# Almacenamiento en memoria
dossier_embeddings = {}
dossier_documents = {} #Trabajamos con un diccionario que va a contener el diccionario de cada chunk referenciado por su id unica
new_documents = {}
new_embeddings = {}

historial_de_chat = []

# Directorio donde se encuentran los archivos JSON
directory_docs = "./resources"
directory_embs = "./embeddings"
directory_new_docs = "./new_resources"
directory_new_embs = "./new_embeddings"

# Si no existen, son creados
os.makedirs(directory_docs, exist_ok=True)
os.makedirs(directory_embs, exist_ok=True)
os.makedirs(directory_new_docs, exist_ok=True)
os.makedirs(directory_new_embs, exist_ok=True)

# Cargar los archivos JSON
documents = load_json_files(directory_docs)
embeddings = load_json_files(directory_embs)
new_documents = load_json_files(directory_new_docs)
new_embeddings = load_json_files(directory_new_embs)

###################################################
#################### Endpoints ####################
###################################################

class Consulta(BaseModel):
    prompt: str

@app.post("/asesoria_legal_dev")
async def chat_dev(query: Consulta):
    return generate_answer(query, developer=True)

@app.post("/asesoria_legal_haiku")
async def chat_haiku(query: Consulta):
	return generate_answer(query, haiku=True)

@app.post("/asesoria_legal")
async def chat_custom(query: Consulta):
	return generate_answer(query)

@app.get("/historial_de_chat")
async def print_chat():
    return historial_de_chat

# Manejo de errores
async def global_exception_handler(request: Request, exc: Exception):
    return {
        "status_code": 500,
        "detail": "Internal Server Error",
        "error": str(exc)
    }