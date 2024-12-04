from ultralytics import YOLO
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import numpy as np

model_people = YOLO("yolov8n.pt")
model_epi = YOLO("datsetpropriov2.pt")
app = FastAPI()

PERSON_CLASS_ID = 0 
HELMET_CLASS_ID = 0
VEST_CLASS_ID = 1
GLOVES_CLASS_ID = 2
MASK_CLASS_ID = 3
GLASSES_CLASS_ID = 4 

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Renderiza a página inicial."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect-epi/")
async def detect_epi(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    np_image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    results_people = model_people(img)
    for result_people in results_people:
        person_boxes = [
            box for box, cls in zip(result_people.boxes.xyxy.cpu().numpy(), 
                                    result_people.boxes.cls.cpu().numpy().astype(int))
            if cls == PERSON_CLASS_ID
        ]

        for person_box in person_boxes:
            x1, y1, x2, y2 = map(int, person_box)
            person_roi = img[y1:y2, x1:x2]

            results_epi = model_epi(person_roi)
            has_helmet = False
            has_glasses = False
            has_gloves = False
            has_mask = False
            has_vest = False
            epi_detected = ""

            for result_epi in results_epi:
                for box, cls in zip(result_epi.boxes.xyxy.cpu().numpy(), 
                                    result_epi.boxes.cls.cpu().numpy().astype(int)):
                    ex1, ey1, ex2, ey2 = map(int, box)
                    if cls == HELMET_CLASS_ID:
                        has_helmet = True
                        cv2.rectangle(person_roi, (ex1, ey1), (ex2, ey2), (255, 255, 0), 2)
                        cv2.putText(
                            person_roi, "Capacete", 
                            (ex1, ey1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
                        )
                        epi_detected += "Capacete;"
                    if cls == VEST_CLASS_ID:
                        has_vest = True
                        cv2.rectangle(person_roi, (ex1, ey1), (ex2, ey2), (255, 0, 0), 2)
                        cv2.putText(
                            person_roi, "Colete", 
                            (ex1, ey1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                        )
                        epi_detected += "Colete;"
                    if cls == GLOVES_CLASS_ID:
                        has_gloves = True
                        cv2.rectangle(person_roi, (ex1, ey1), (ex2, ey2), (0, 255, 255), 2)
                        cv2.putText(
                            person_roi, "Luvas", 
                            (ex1, ey1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
                        )
                        epi_detected += "Luvas;"
                    if cls == MASK_CLASS_ID:
                        has_mask = True
                        cv2.rectangle(person_roi, (ex1, ey1), (ex2, ey2), (255, 0, 255), 2)
                        cv2.putText(
                            person_roi, "Mascara", 
                            (ex1, ey1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2
                        )
                        epi_detected += "Máscara;"
                    if cls == GLASSES_CLASS_ID:
                        has_glasses = True
                        cv2.rectangle(person_roi, (ex1, ey1), (ex2, ey2), (0, 0, 0), 2)
                        cv2.putText(
                            person_roi, "Oculos", 
                            (ex1, ey1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                        )
                        epi_detected += "Óculos;"
            label = "Com EPI" if has_helmet or has_glasses or has_mask or has_gloves or has_vest else "Sem EPI"
            color = (0, 255, 0) if has_helmet or has_glasses or has_mask or has_gloves or has_vest else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, label, 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
            )
    cv2.imshow("Detecção de Pessoas e EPIs", img)
    #return {"status": "success", "epi_detected": epi_detected, "img": img}
    if epi_detected == "":
        epi_detected = "Nenhum EPI foi detectado"
        
    return templates.TemplateResponse(
    "index.html", 
    {"request": request, "status": "success", "epi_detected": epi_detected, "img": img})