from ultralytics import YOLO
import cv2

model = YOLO("./best.pt")
funil_box = None 

def processar_frame(frame):
    global funil_box

    resultados = model(frame)
    camioes_abaixo_funil = []

    for result in resultados:
        for det in result.boxes.data.tolist():
            confidence = det[4]
            classe_id = int(det[5])
            nome_classe = model.names[classe_id]

            if confidence > 0.5:
                x, y, w, h = det[:4]
                box = (x, y, w - x, h - y)

                if nome_classe == "grua":
                    if grua_no_funil(box):
                        print("Grua acima do funil.")
                elif nome_classe == "funil":
                    funil_box = box
                    print("Funil detectado.")
                elif nome_classe == "camiao":
                    if camiao_abaixo_funil(box):
                        print("Caminhão abaixo do funil.")
                        camioes_abaixo_funil.append(box)
                    else:
                        print("Caminhão ignorado, não abaixo do funil.")

    if camioes_abaixo_funil:
        for camiao in camioes_abaixo_funil:
            x, y, w, h = camiao
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(frame, "Caminhao", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)  

def camiao_abaixo_funil(camiao_box, distancia_minima=50):
    global funil_box
    if funil_box is None:
        return False

    camiao_x, camiao_y, camiao_w, camiao_h = camiao_box
    funil_x, funil_y, funil_w, funil_h = funil_box

    return (camiao_y + camiao_h >= funil_y + distancia_minima) and (
        camiao_x + camiao_w > funil_x and camiao_x < funil_x + funil_w
    )

def grua_no_funil(grua_box):
    global funil_box
    if funil_box is None:
        return False
    grua_x, grua_y, grua_w, grua_h = grua_box
    funil_x, funil_y, funil_w, funil_h = funil_box
    return (grua_x > funil_x and grua_x + grua_w < funil_x + funil_w and grua_y + grua_h > funil_y)

def processar_video(caminho_video):
    cap = cv2.VideoCapture(caminho_video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (320, 240))  
        processar_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

processar_video("./teste.mp4")
