import pickle
import uvicorn
import numpy as np
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from experiment.baseline_with_samples import train

app = FastAPI()

@app.post("/")
async def upload_file(file: UploadFile = File(...)):
    save_path = Path("data.pkl")
    content = await file.read()

    with open(save_path, "wb") as f:
        f.write(content)

    try:
        # pickle load data
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        train(config="config/experiment/baseline_with_samples.yaml", samples=data)

    except Exception as e:
        return {"status": "error", "message": f"读取 data.npy 失败: {e}"}

    return FileResponse(path="export_models/model.pth")

if __name__ == "__main__":
    # with open("data.pkl", "rb") as f:
    #     data = pickle.load(f)
    # train(config="config/experiment/baseline_with_samples.yaml", samples=data)
    uvicorn.run(app, host="0.0.0.0", port=37131)