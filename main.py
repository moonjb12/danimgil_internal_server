from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from yolov5 import detect_module
import os

app = FastAPI()

result_li = ['Animals(Dolls)', 'Person', 'Garbage bag & sacks', 'Construction signs & Parking prohibited board', 'Traffic cone', 'Box', 'Stones on road', 'Pothole on road', 'Filled pothole', 'Manhole']


@app.get("/")
async def root():
    return "This is admin's page! Why are you here?"


@app.post("/public")
async def upload_photo(file: UploadFile, uuid: str):
    UPLOAD_DIR = f"C:\\Users\\JBMOON\\Desktop\\DaNimGil\\public\\{uuid}"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    content = await file.read()
    filename = f"{uuid}.png"
    print(os.path)
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
        fp.write(content)

    res = int(detect_module.run(weights="best.pt", source=f"./public/{uuid}/{uuid}.png", project=f"./public/{uuid}", name="detect").split()[0])
    
    return {"filename": file.filename, "uuid": uuid, "result": result_li[res]}


@app.get("/download")
async def download_image(uuid: str):
    file_path = f"C:\\Users\\JBMOON\\Desktop\\DaNimGil\\public\\{uuid}\\detect\\{uuid}.png"

    if not os.path.exists(file_path):
        return "This picture is not detected yet"

    return FileResponse(
        path=file_path,
        filename=f"{uuid}.png"
    )
