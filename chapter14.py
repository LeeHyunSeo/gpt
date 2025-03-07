from typing import Any, Dict
from fastapi import Body, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title = "Jaecolacus Maximus Quote Giver",
    description = "Get a real quote said by Jaecolacus Maximus himself.",
    servers=[{
        "url": "https://exit-outlets-ac-animal.trycloudflare.com"
    }]
)

class Quote(BaseModel):
    quote: str = Field(description="The quote that Jaecolacus Maximus said.")
    year: int = Field(description="The year when Jaecolacus Maximus said the quote.")

@app.get(
        "/quote", 
        summary="Returns a random quote by Jaecolacus Maximus",
        description="Upon receiving a GET request this endpoint will return a real quiote said by Jaecolacus Maximus himself.",
        response_description="A Quote object that contains the quote said by Jaecolacus Maximus and the data when the quote was said.",
        response_model=Quote
         )
def get_quote(request: Request):
    print(request.headers)
    return {
        "quote": "Life is short so eat it all",
        "year" : 1950
    }

user_token_db = {
    "ABCDEF": "jaewon"
}

@app.get("/authorize", response_class=HTMLResponse)
def handel_authorize(client_id: str, redirect_uri:str, state: str):
    return f"""
    <html>
        <head>
            <title>Nicolacus Maximus Log In</title>
        </head>
        <body>
            <h1>Log Into Nicolacus Maximus</h1>
            <a href="{redirect_uri}?code=ABCDEF&state={state}">Authorize Nicolacus Maximus GPT</a>
        </body>
    </html>
    """

@app.post("/token")
def handle_token(code = Form(...)):
    print(code)
    return {"access_token":user_token_db[code]}
