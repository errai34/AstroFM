import os
import logging
from typing import Optional, List, Tuple

import openai
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ingest import ingest_docs
from langchain.vectorstores import VectorStore
from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse

# Set up environment and OpenAI API key
os.environ["LANGCHAIN_HANDLER"] = "langchain"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None


# Dependency for loading VectorStore
async def get_vectorstore() -> VectorStore:
    global vectorstore
    if vectorstore is None:
        logging.info("Loading VectorStore")
        vectorstore = ingest_docs()
    return vectorstore


# Serve main page
@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# WebSocket chat endpoint
@app.websocket("/chat")
async def websocket_endpoint(
    websocket: WebSocket, vectorstore: VectorStore = Depends(get_vectorstore)
):
    await websocket.accept()

    # Initialize handlers and chat history
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history: List[Tuple[str, str]] = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=False)

    while True:
        try:
            # Receive question from client
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Send start response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())
            logging.info(f"start_resp={start_resp.dict()}")

            # Generate and send answer
            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result["answer"]))

            # Send end response
            end_resp = ChatResponse(sender="bot", message="", type="end")
            logging.info(f"end_resp={end_resp.dict()}")
            await websocket.send_json(end_resp.dict())

        except WebSocketDisconnect:
            logging.info("WebSocket disconnected")
            break

        except Exception as e:
            logging.error(f"Error occurred: {e}")
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())
            raise e


# Run app using uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
