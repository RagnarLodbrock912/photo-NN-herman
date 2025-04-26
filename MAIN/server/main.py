from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
from io import BytesIO
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from models import filterData
from child_process import run_child_process
import json

templates = Jinja2Templates(directory="templates")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

INPUT_IMAGE_PATH = "input_images/"
OUTPUT_IMAGE_PATH = "static/images/"

@app.post("/process-image")
async def process_image(filter: str = Form(...), file: UploadFile = File(...)):
    parsed_filter = filterData.model_validate_json(filter)

    image_path = INPUT_IMAGE_PATH + file.filename
    
    contents = await file.read()

    with open(image_path, 'wb') as f:
        f.write(contents)

    stdin = f"{len(parsed_filter.filter)}\n{"\n".join([" ".join(map(str, _list)) for _list in parsed_filter.filter])}\n{file.filename}\n"

    run_child_process(
        _CMD="../NN/main",
        _stdin=stdin
    )

    return {
        "detail": "success",
        "image_url": OUTPUT_IMAGE_PATH + file.filename
    }
    
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)