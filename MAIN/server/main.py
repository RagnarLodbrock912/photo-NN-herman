from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()

def get_rgb_channels(image: Image.Image):
    image_rgb = image.convert('RGB')
    np_image = np.array(image_rgb)
    
    red_channel = np_image[:,:,0]
    green_channel = np_image[:,:,1]
    blue_channel = np_image[:,:,2]
    
    return red_channel, green_channel, blue_channel

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))
    
    red_channel, green_channel, blue_channel = get_rgb_channels(image)
    print(red_channel.tolist())
    print('-' * 100)
    print(green_channel.tolist())
    print('-' * 100)
    print(blue_channel.tolist())

    return {
        "filename": file.filename, 
        "red_channel": len(red_channel.tolist()), 
        "green_channel": len(green_channel.tolist()),
        "blue_channel": len(blue_channel.tolist())
    }



if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)