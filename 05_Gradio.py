# Importazione delle librerie standard di Python
import os  # Per accedere alle variabili d'ambiente
from random import randrange  # Per generare numeri casuali per l'ID di sessione
import time  # Per ottenere timestamp

# Importazione delle librerie LangChain per la gestione del modello di chat e delle conversazioni
from langchain_openai import ChatOpenAI  # Classe per interfacciarsi con il modello GPT di OpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder  # Per creare template di prompt strutturati
from langchain.schema import SystemMessage, HumanMessage  # Classi per rappresentare messaggi di sistema e utente
from langchain_core.chat_history import InMemoryChatMessageHistory  # Gestione della cronologia chat in memoria
from langchain_core.runnables.history import RunnableWithMessageHistory  # Wrapper per aggiungere cronologia a una chain
from langchain.prompts import MessagesPlaceholder  # Importazione duplicata (già importata sopra)

# Importazione di Gradio per creare l'interfaccia web
import gradio as gr # pip install gradio

# Importazione per gestire le variabili d'ambiente dal file .env
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env (come la chiave API di OpenAI)
load_dotenv()

# Setup cartella logs e path file summary
os.makedirs("logs", exist_ok=True)
SUMMARIES_LOG_PATH = os.path.join("logs", "summaries.log")

# Inizializzazione del modello di chat OpenAI con parametri personalizzati
llm = ChatOpenAI(
    openai_api_key=os.getenv("openai_api_key"),     # Chiave API recuperata dalle variabili d'ambiente
    temperature=.75,                                # Controlla la creatività del modello (0 = deterministica, 1 = molto creativa)
    max_tokens=1024,                                # Limite massimo di token per risposta
    request_timeout=30                              # Timeout per le richieste in secondi
)

# Template del messaggio di sistema che definisce il comportamento del chatbot
system_template = (
    "Act like a useful assistant and answer the user questions using the information the user gives to you during the conversation."
)

# Creazione del template di prompt che include:
# - Un messaggio di sistema fisso
# - Un placeholder per la cronologia dei messaggi
# - Un placeholder per l'input dell'utente
prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),  # Messaggio di sistema che definisce il ruolo del bot
    MessagesPlaceholder(variable_name="history"),  # Placeholder per i messaggi precedenti della conversazione
    ("user", "{input}"),  # Placeholder per l'input attuale dell'utente
])

# Creazione della chain LangChain: prompt + modello LLM
# Il simbolo | rappresenta una pipeline: prompt viene passato al modello llm
chain = prompt | llm

# Dizionario globale per memorizzare le cronologie di chat per ogni sessione
# Chiave: session_id, Valore: InMemoryChatMessageHistory
store = {}

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Recupera o crea una nuova cronologia di chat per una sessione specifica.
    
    Args:
        session_id (str): Identificatore univoco della sessione
        
    Returns:
        InMemoryChatMessageHistory: Oggetto che mantiene la cronologia dei messaggi
    """
    # Se la sessione non esiste ancora, crea una nuova cronologia vuota
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrapper che aggiunge la gestione della cronologia alla chain
# Questo permette al modello di "ricordare" le conversazioni precedenti
chain_with_history = RunnableWithMessageHistory(
    chain,  # La chain base (prompt + modello)
    get_session_history=get_chat_history,  # Funzione per recuperare la cronologia
    input_messages_key="input",  # Nome della chiave per l'input utente nel dizionario
    history_messages_key="history"  # Nome della chiave per la cronologia nel template
)

# Helper per contare i messaggi totali e formattare un blocco di testo ---
def _count_messages(history_list):
    return len(history_list or [])

def _build_text_window(history_list, window=10):
    """Prepara il testo da riassumere prendendo gli ultimi 'window' messaggi."""
    if not history_list:
        return ""
    slice_ = history_list[-window:]
    # history è una lista di dict: {"role": "...", "content": "..."}
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in slice_)


def stream_response(session_id, input_text, history):
    """
    Funzione che gestisce lo streaming della risposta del chatbot.
    Questa funzione è un generatore che produce output incrementali per aggiornare
    l'interfaccia Gradio in tempo reale mentre il modello genera la risposta.
    
    Args:
        session_id: ID univoco della sessione corrente
        input_text (str): Testo inserito dall'utente
        history (list): Lista dei messaggi precedenti della conversazione
        
    Yields:
        tuple: (session_id, messaggio_vuoto, cronologia_aggiornata)
    """
    
    # Controllo se l'input è vuoto o contiene solo spazi
    if not input_text or len(input_text.strip()) == 0:
        yield session_id, "", history  # Restituisce i valori invariati
        return

    # Inizializza la cronologia se è None (prima conversazione)
    if history is None:
        history = []

    # Variabile per accumulare la risposta parziale del modello
    partial_response = ""

    try:
        # Avvia lo streaming della risposta dal modello
        # stream() restituisce un generatore che emette token man mano che vengono generati
        stream = chain_with_history.stream(
            {"input": input_text},  # Input dell'utente
            config={"session_id": str(session_id)}  # Configurazione con l'ID di sessione
        )

        # Itera su ogni token generato dal modello
        for token in stream:
            delta = token.content  # Estrae il contenuto del token corrente
            if delta:  # Solo se il token contiene effettivamente del testo
                partial_response += delta  # Accumula il token alla risposta parziale

                # Costruisce una cronologia temporanea per l'aggiornamento dell'UI
                # Include tutta la conversazione precedente + il nuovo scambio in corso
                temp_history = history + [
                    {"role": "user", "content": input_text},  # Messaggio dell'utente corrente
                    {"role": "assistant", "content": partial_response}  # Risposta parziale del bot
                ]

                # Yield intermedio per aggiornare l'interfaccia Gradio in tempo reale
                # Il secondo parametro ("") svuota il campo di input cioè la TextBox
                yield session_id, "", temp_history

        # Attenzione: questa history è per Gradio!
        # Una volta completato lo streaming, aggiorna la cronologia definitiva
        # Questo serve per mantenere la conversazione per le interazioni successive
        history.append({"role": "user", "content": input_text})
        history.append({"role": "assistant", "content": partial_response})

        # Riassunto ogni 10 messaggi totali (user+assistant) ---
        try:
            total_msgs = _count_messages(history)
            if total_msgs % 10 == 0:
                # Costruiamo una finestra compatta degli ultimi 10 messaggi
                text_window = _build_text_window(history, window=10)

                summary_res = llm.invoke([
                    SystemMessage(
                        content=(
                            "Produce a very short summary prefixed with 'Summary:'. "
                            "Be concise."
                        )
                    ),
                    HumanMessage(content=text_window)
                ])

                # Scrive su file con timestamp e session id
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                log_block = (
                    f"[{ts}] session_id={session_id}\n"
                    f"{summary_res.content}\n"
                    f"{'-'*60}\n"
                )
                with open(SUMMARIES_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(log_block)

                # (Facoltativo) stampa su console per debug
                print(f"\n--- Saved 10-message summary for session {session_id} ---\n")
        except Exception as e:
            print(f"Errore nella generazione/salvataggio del riassunto: {e}")

    except Exception as e:
        # Gestisce errori nello streaming principale
        print(f"Errore nello streaming: {e}")
        # Anche in caso di errore, salva quello che è stato generato finora (anche questa history è per Gradio!)
        history.append({"role": "user", "content": input_text})
        history.append({"role": "assistant", "content": partial_response})
        yield session_id, "", history


# Creazione dell'interfaccia Gradio
with gr.Blocks(css="""footer{display:none !important}""") as demo:  # CSS per nascondere il footer
    
    # Stato che mantiene l'ID di sessione univoco per ogni utente
    # L'ID è formato da: numero_casuale_timestamp_numero_casuale per garantire unicità
    session_id_state = gr.State(value=lambda: f"{randrange(10000, 99999)}_{int(time.time() * 1000)}_{randrange(10000, 99999)}")
    
    # Componente chatbot che visualizza la conversazione
    # type="messages" specifica il formato dei messaggi
    chatbot = gr.Chatbot(type="messages")
    
    # Campo di input per i messaggi dell'utente
    msg = gr.Textbox(placeholder="Scrivi qui il tuo messaggio", label="")

    # Collega la funzione di streaming all'evento submit (invio) del campo di input
    # Quando l'utente preme Invio o clicca submit:
    # - Passa: [session_id, messaggio_utente, cronologia_corrente]
    # - Riceve: [session_id_aggiornato, campo_input_svuotato, cronologia_aggiornata]
    msg.submit(
        stream_response,                    # Funzione da chiamare 
        [session_id_state, msg, chatbot],   # Input: stato sessione, messaggio, chatbot
        [session_id_state, msg, chatbot]    # Output: aggiorna gli stessi componenti
    )

# Avvia l'applicazione Gradio
# queue() abilita la gestione delle code per le richieste multiple
# debug=True abilita modalità debug
# share=True crea un link pubblico temporaneo per condividere l'app
demo.queue().launch(debug=True, share=True)
