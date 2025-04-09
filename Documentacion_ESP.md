# Documentación para `/C:/Users/Mara/Desktop/ChallengeFinal/main.py`

## Descripción General
El script es una aplicación de FastAPI diseñada para proporcionar asesoramiento legal mediante:
1. Un procesamiento inicial de documentos de jurisprudencia y dogma, la generación de embeddings y la búsqueda de información legal relevante para realizar una respuesta a la consulta del usuario.
2. Una búsqueda en internet de las leyes referidas en la primera respuesta para ampliar la respuesta con posibles artículos de relevancia y su procesamiento. (Excepto si `developer==False` en la función principal, i.e. endpoint `asesoria_legal_dev`).
3. Una curaduría de la respuesta inicial con la información legal ampliada por el web scraping.

Integra varias funcionalidades como el procesamiento de PDF, la fragmentación de documentos, la generación de embeddings y el web scraping para mejorar el proceso de asesoramiento legal.

## Importaciones
El script importa varias bibliotecas y módulos, incluyendo:

- `FastAPI` para crear la aplicación web.
- `pydantic` para la validación de datos.
- `cohere` para generar embeddings.
- `sklearn` para calcular la similitud coseno.
- `dotenv` para cargar variables de entorno.
- [`os`](file:///c:/Program%20Files/WindowsApps/PythonSoftwareFoundation.Python.3.12_3.12.2288.0_x64__qbz5n2kfra8p0/Lib/os.py) para interactuar con el sistema de archivos.
- `pypdf` para leer archivos PDF.
- `requests` y `BeautifulSoup` para el web scraping.
- [`uuid`](file:///c:/Program%20Files/WindowsApps/PythonSoftwareFoundation.Python.3.12_3.12.2288.0_x64__qbz5n2kfra8p0/Lib/uuid.py) para generar identificadores únicos.
- [`json`](file:///c:/Program%20Files/WindowsApps/PythonSoftwareFoundation.Python.3.12_3.12.2288.0_x64__qbz5n2kfra8p0/Lib/json/__init__.py) para manejar datos JSON.

## Funciones

### Procesamiento de PDF
- `load_pdf(path_pdf)`: Carga el contenido de un archivo PDF desde la ruta dada y devuelve el texto extraído.
- `histories_splitter(historias, chunk_size)`: Divide el texto proporcionado en fragmentos más pequeños del tamaño especificado y devuelve una lista de cadenas.
- `splitt(content: str)`: Divide el contenido en fragmentos de aproximadamente 2300 caracteres con un solapamiento del 10% y devuelve una lista de cadenas.

### Almacenamiento de Documentos
- `save_doc(input_doc, save_local=True)`: Guarda el texto del documento proporcionado en archivos JSON y los almacena en el diccionario `documents`. Opcionalmente guarda los archivos localmente.
- `save_new_doc(input_doc, save_local=True)`: Similar a `save_doc`, pero almacena los documentos en el diccionario `new_documents` y los guarda en el directorio `./new_resources`.

### Generación de Embeddings
- `embedd_doc(doc_id: str, save_local=True)`: Genera embeddings para el documento con el ID dado y los almacena en el diccionario `embeddings`. Opcionalmente guarda los embeddings localmente.
- `embedd_new_doc(doc_id: str, save_local=True)`: Genera embeddings para el documento con el ID dado en el diccionario `new_documents` y los almacena en el diccionario `new_embeddings`. Opcionalmente guarda los embeddings localmente.
- `embedd_query(query: str)`: Genera un embedding para la cadena de consulta dada y lo devuelve.

### Manejo de Archivos JSON
- `load_json_files(directory)`: Carga todos los archivos JSON del directorio especificado y devuelve un diccionario con los datos cargados.

### Búsqueda de Documentos
- `search_documents(query_embedding, embeds, docs, top_n=5)`: Busca los documentos más similares al embedding de consulta dado en los embeddings y documentos proporcionados. Devuelve los N documentos más similares.

### Web Scraping
- `extraer_primer_resultado(html)`: Extrae el enlace al primer resultado del contenido HTML dado.
- `extraer_texto_norma(url)`: Extrae el contenido textual de un documento legal desde la URL dada.
- `procesar_nueva_norma(texto: str)`: Procesa un nuevo documento legal guardándolo y generando embeddings para él.

### Evaluación de Fidelidad
- `get_faithfulness(prompt, respuesta, contexto)`: Evalúa la fidelidad de una respuesta con respecto al contexto proporcionado.

### Integración con LLM
- `prelude_jurisprudencia()`: Devuelve el preludio para el prompt de jurisprudencia del LLM.
- `prompt_jurisprudencia(docs, query)`: Formatea el prompt para el LLM de jurisprudencia.
- `prelude_web_scraping()`: Devuelve el preludio para el prompt de web scraping del LLM.
- `prompt_advisory(scrape, response_jurisprudencia)`: Formatea el prompt para el LLM de asesoramiento.
- `llm_jurisprudencia(historial_de_chat, query, embeddings, documents, modelo="command-r-plus-08-2024")`: Consulta el LLM de jurisprudencia con el historial de chat, la consulta, los embeddings y los documentos proporcionados. Devuelve la respuesta y el historial de chat actualizado.
- `llm_web_scraper(prompt_ws, developer=True)`: Consulta el LLM de web scraping con el prompt proporcionado. Opcionalmente procesa los resultados en modo desarrollador.
- `llm_advisory(historial_de_chat, response_jurisprudencia, scrape, haiku=False)`: Consulta el LLM de asesoramiento con el historial de chat, la respuesta de jurisprudencia y los datos extraídos. Devuelve la respuesta curada y el historial de chat actualizado.

### Función Principal
- `generate_answer(prompt: str, developer=False, haiku=False)`: Función principal que integra el ciclo completo de la arquitectura RAG. Genera una respuesta de asesoramiento legal basada en el prompt proporcionado.

## Endpoints de FastAPI
- `/asesoria_legal_dev`: Endpoint POST para generar asesoramiento legal en modo desarrollador.
- `/asesoria_legal_haiku`: Endpoint POST para generar asesoramiento legal en modo haiku.
- `/asesoria_legal`: Endpoint POST para generar asesoramiento legal personalizado.
- `/historial_de_chat`: Endpoint GET para recuperar el historial de chat.
- `/fidelidad_de_las_respuestas`: Endpoint GET para recuperar las puntuaciones de fidelidad de las respuestas.

## Manejo de Errores
- `global_exception_handler(request: Request, exc: Exception)`: Maneja excepciones globales y devuelve una respuesta de error interno del servidor 500.

## Declaraciones de Tipos
- `Consulta`: Modelo Pydantic para validar el prompt de entrada.
