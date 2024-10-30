# ylow

1. Importação de Bibliotecas

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import cv2

YOLO: Uma biblioteca para detecção de objetos em tempo real.
DeepSort: Uma implementação do algoritmo Deep SORT, que é usado para rastreamento de objetos.
time: Usado para medir o tempo, útil para calcular a duração em que os objetos estão visíveis.
cv2: A biblioteca OpenCV para manipulação de vídeos e imagens.

2.  Inicialização de Modelos
model = YOLO("./runs/detect/train/weights/best.pt")
tracker = DeepSort(max_age=30, n_init=3)

O modelo YOLO é carregado a partir de um arquivo de pesos, que deve ter sido previamente treinado para detectar as classes de interesse (neste caso, "caminhão" e "grua").
O rastreador Deep SORT é inicializado com dois parâmetros:

    max_age: Define quantos frames o rastreador irá manter um objeto "perdido" antes de descartá-lo.
    n_init: O número mínimo de frames que o rastreador deve ver um objeto para confirmá-lo como um objeto rastreável.

3. Variáveis de Controle

caminhao_id_atual = None
viagens = 0
tempo_total_viagens = 0
tempo_total_grua = 0
inicio_grua = None
inicio_funil = None
funil_box = None  # Inicializa o bounding box do funil

Variáveis de Estado:

    caminhao_id_atual: Armazena o ID do caminhão que está atualmente sob o funil.
    viagens: Contador de quantas viagens o caminhão fez.
    tempo_total_viagens: Total de tempo que o caminhão ficou sob o funil.
    tempo_total_grua: Total de tempo que a grua esteve visível.
    inicio_grua e inicio_funil: Usadas para armazenar o tempo de início da visibilidade da grua e do caminhão sob o funil.
    funil_box: Armazena as coordenadas do bounding box do funil.

4. Função caminhao_esta_debaixo_funil

def caminhao_esta_debaixo_funil(caminhao_box):
    if funil_box is None:
        return False
    caminhao_x, caminhao_y, caminhao_w, caminhao_h = caminhao_box
    funil_x, funil_y, funil_w, funil_h = funil_box
    return (
        caminhao_x > funil_x and caminhao_x + caminhao_w < funil_x + funil_w and
        caminhao_y + caminhao_h > funil_y
    )

Esta função verifica se o caminhão está posicionado dentro da área do funil.
Compara as coordenadas do caminhão (bounding box) com as do funil.
Retorna True se o caminhão estiver debaixo do funil e False caso contrário.

5. Função processar_frame

def processar_frame(frame):
    global caminhao_id_atual, viagens, tempo_total_viagens, tempo_total_grua, inicio_grua, inicio_funil, funil_box

    resultados = model(frame)

    deteccoes = []
    for r in resultados:
        for det in r.boxes.data.tolist():
            x, y, w, h = det[:4]
            confidence = det[4]
            classe_id = int(det[5])
            nome_classe = model.names[classe_id]
            
            if confidence > 0.5:  # Ajuste conforme necessário
                deteccoes.append(([x, y, w - x, h - y], confidence, nome_classe))
                
                if nome_classe == "funil":
                    funil_box = (x, y, w - x, h - y)


Detecção de Objetos

    YOLO é usado para detectar objetos no frame atual.
    Os resultados contêm informações sobre as caixas delimitadoras (bounding boxes) e a classe dos objetos detectados.
    Para cada detecção, se a confiança for superior a 0.5, a detecção é armazenada na lista deteccoes.

Atualização do Bounding Box do Funil

    Se a classe detectada for "funil", atualiza funil_box com as coordenadas do bounding box do funil.

Rastreamento com Deep SORT

    trackings = tracker.update_tracks(deteccoes, frame=frame)

As detecções são passadas para o rastreador Deep SORT, que atualiza o estado dos objetos rastreados.

Gerenciamento do Caminhão e Gruas

    ids_ativos = set()  # Para rastrear IDs ativos de gruas

    for track in trackings:
        if track.state == "confirmed":
            track_id = track.track_id
            classe = track.det_class
            bbox = track.to_tlbr()

            if classe == "caminhao" and funil_box and caminhao_esta_debaixo_funil(bbox):
                if caminhao_id_atual is None:
                    caminhao_id_atual = track_id
                elif caminhao_id_atual != track_id:
                    print("Caminhão trocado! Reiniciando a contagem de viagens.")
                    caminhao_id_atual = track_id
                    viagens = 0


    Para cada objeto rastreado, se seu estado for "confirmado", o ID do rastreador, a classe e o bounding box são obtidos.
    Verifica se é um caminhão e se está sob o funil:
        Se for o primeiro caminhão detectado, atualiza caminhao_id_atual.
        Se um caminhão diferente for detectado, reinicia a contagem de viagens.


Contabilização do Tempo da Grua

            elif classe == "grua":
                if inicio_grua is None:
                    inicio_grua = time.time()
                
                ids_ativos.add(track_id)

                if caminhao_id_atual and caminhao_esta_debaixo_funil(bbox):
                    if inicio_funil is None:
                        inicio_funil = time.time()  # Começa a contagem sobre o funil
                else:
                    if inicio_funil is not None:
                        tempo_funil = time.time() - inicio_funil
                        tempo_total_viagens += tempo_funil
                        print(f"Tempo da grua sobre o funil: {tempo_funil:.2f} segundos")
                        inicio_funil = None  # Reinicia o tempo da grua sobre o funil


Para a grua, se ela for detectada e inicio_grua estiver vazio, armazena o tempo atual.
Se um caminhão está sob o funil, começa a contar o tempo que a grua está sobre ele.
Se a grua não está mais sobre o funil, calcula o tempo e atualiza o total de viagens.

Tempo Total da Grua

            if track_id in ids_ativos and track.state == "lost":
                if inicio_grua is not None:
                    tempo_grua = time.time() - inicio_grua
                    tempo_total_grua += tempo_grua
                    print(f"Tempo total da grua visível: {tempo_grua:.2f} segundos")
                    inicio_grua = None  # Reinicia o tempo total

Se a grua "perde" o rastreamento, calcula o tempo total que ela esteve visível e atualiza tempo_total_grua.

6. Função processar_video

def processar_video(caminho_video):
    cap = cv2.VideoCapture(caminho_video)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))  # Ajuste a resolução conforme necessário

        frame_count += 1
        
        if frame_count % 2 == 0:
            processar_frame(frame)

            if frame_count % 5 == 0:
                cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

Captura de Vídeo: Abre o vídeo e lê os frames em um loop.
Redução de Resolução: Cada frame é redimensionado para melhorar a eficiência do processamento.
Processamento a Cada Dois Frames: Chama processar_frame para cada segundo frame.
Exibição de Frame: A cada cinco frames, exibe o frame processado.
Saída do Loop: O loop pode ser interrompido pressionando a tecla 'q'.

7. Chamada da Função Principal

processar_video("./teste.mp4")


Geral

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import cv2

# Carregar o modelo YOLO treinado
model = YOLO("./runs/detect/train/weights/best.pt")

# Inicializar o rastreador Deep SORT
tracker = DeepSort(max_age=30, n_init=3)

# Variáveis de controle para o caminhão e contagem de viagens
caminhao_id_atual = None
viagens = 0
tempo_total_viagens = 0
tempo_total_grua = 0
inicio_grua = None
inicio_funil = None
funil_box = None  # Inicializa o bounding box do funil

# Função para verificar se o caminhão está debaixo do funil
def caminhao_esta_debaixo_funil(caminhao_box):
    if funil_box is None:
        return False
    caminhao_x, caminhao_y, caminhao_w, caminhao_h = caminhao_box
    funil_x, funil_y, funil_w, funil_h = funil_box
    return (
        caminhao_x > funil_x and caminhao_x + caminhao_w < funil_x + funil_w and
        caminhao_y + caminhao_h > funil_y
    )

# Função para detectar e rastrear objetos
def processar_frame(frame):
    global caminhao_id_atual, viagens, tempo_total_viagens, tempo_total_grua, inicio_grua, inicio_funil, funil_box

    # Fazer detecção com YOLO
    resultados = model(frame)

    # Converter resultados para Deep SORT
    deteccoes = []
    for r in resultados:
        for det in r.boxes.data.tolist():
            x, y, w, h = det[:4]
            confidence = det[4]
            classe_id = int(det[5])
            nome_classe = model.names[classe_id]
            
            # Filtra por confiança
            if confidence > 0.5:  # Ajuste conforme necessário
                deteccoes.append(([x, y, w - x, h - y], confidence, nome_classe))
                
                # Verifica se a classe detectada é o funil
                if nome_classe == "funil":
                    funil_box = (x, y, w - x, h - y)

    # Rastreamento com Deep SORT
    trackings = tracker.update_tracks(deteccoes, frame=frame)

    # Identificar caminhão e gerenciar contagem
    ids_ativos = set()  # Para rastrear IDs ativos de gruas

    for track in trackings:
        if track.state == "confirmed":
            track_id = track.track_id
            classe = track.det_class
            bbox = track.to_tlbr()

            # Verifica se o caminhão está debaixo do funil
            if classe == "caminhao" and funil_box and caminhao_esta_debaixo_funil(bbox):
                if caminhao_id_atual is None:
                    caminhao_id_atual = track_id
                elif caminhao_id_atual != track_id:
                    print("Caminhão trocado! Reiniciando a contagem de viagens.")
                    caminhao_id_atual = track_id
                    viagens = 0  # Reset apenas se for um caminhão diferente

            elif classe == "grua":
                # Contabiliza o tempo total que a grua está visível
                if inicio_grua is None:
                    inicio_grua = time.time()
                
                # Adiciona o ID ativo da grua
                ids_ativos.add(track_id)
                
                # Verifica se a grua está sobre o funil
                if caminhao_id_atual and caminhao_esta_debaixo_funil(bbox):
                    if inicio_funil is None:
                        inicio_funil = time.time()  # Começa a contagem sobre o funil
                else:
                    # Se a grua não está mais sobre o funil, conta a viagem
                    if inicio_funil is not None:
                        tempo_funil = time.time() - inicio_funil
                        tempo_total_viagens += tempo_funil
                        print(f"Tempo da grua sobre o funil: {tempo_funil:.2f} segundos")
                        inicio_funil = None  # Reinicia o tempo da grua sobre o funil

            # Se a grua desaparece da tela, registra o tempo total
            if track_id in ids_ativos and track.state == "lost":
                if inicio_grua is not None:
                    tempo_grua = time.time() - inicio_grua
                    tempo_total_grua += tempo_grua
                    print(f"Tempo total da grua visível: {tempo_grua:.2f} segundos")
                    inicio_grua = None  # Reinicia o tempo total

# Função principal para ler vídeo e processar cada frame
def processar_video(caminho_video):
    cap = cv2.VideoCapture(caminho_video)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Reduzir a resolução do frame para aumentar a velocidade
        frame = cv2.resize(frame, (320, 240))  # Ajuste a resolução conforme necessário

        frame_count += 1
        
        # Processar apenas a cada dois quadros
        if frame_count % 2 == 0:
            processar_frame(frame)

            # Atualiza a visualização a cada 5 quadros
            if frame_count % 5 == 0:
                cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Chamar a função principal com o caminho do vídeo
processar_video("./teste.mp4")

