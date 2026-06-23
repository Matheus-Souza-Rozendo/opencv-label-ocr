import itertools
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

#=======================================================================================
#Definição das constantes do sistema
LARGURA_MAXIMA = 834
ALTURA_MAXIMA = 382
MIN_AREA = 155
LETRAS = "ABCDEFGHIJKLMNOPQRSTUVWXZ"

#========================================================================================
# FUNÇÕES AUXILIARES

def gerar_mascara(imagem):
    cor_referencia = np.array([106,180,120])
    return cv2.inRange(imagem, cor_referencia, cor_referencia)

def gerar_bordas(imagem):
    bordas = cv2.Canny(imagem, 100, 255)
    kernel = np.ones((3,3),np.uint8)
    return cv2.dilate(bordas, kernel, iterations = 1)


def aproxima_contorno(contorno):
    perimetro = cv2.arcLength(contorno, True)
    epsilon = 0.03 * perimetro
    return cv2.approxPolyDP(contorno, epsilon, True)

def calcular_centroide(imagem,contorno)-> dict:
    M = cv2.moments(contorno)
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(imagem, (cX, cY), 7, (255, 0, 0), -1)
        return {
            "x":cX,
            "y":cY
        }

def transformar_etiqueta(imagem,origem,destino):
    M = cv2.getPerspectiveTransform(origem.astype("float32"), destino)
    return cv2.warpPerspective(imagem, M, (LARGURA_MAXIMA, ALTURA_MAXIMA))

def agrupar_contornos_por_linha(contornos, limiar_distancia=100):
    linhas = []
    for _, grupo in itertools.groupby(contornos, key=lambda c: cv2.boundingRect(c)[1] // limiar_distancia):
        linhas.append(sorted(list(grupo), key=lambda c: cv2.boundingRect(c)[0]))
    return linhas

def formatar_saida(texto:str)->str:
    linhas =  texto.split("\n")
    linhas[0] = linhas[0][:6] + ": " + linhas[0][6:-1] + " " + linhas[0][-1]
    linhas[1] = linhas[1][:7] + ": " + linhas[1][7:-1] + " " + linhas[1][-1]
    linhas[2] = linhas[2][:7] + ": " + linhas[2][7:]
    return f'{linhas[0]}\n{linhas[1]}\n{linhas[2]}'
# ==============================================================================================

def encontrar_etiqueta(imagem):

    #gerando a mascara para identificar apenas as formas no canto da etiqueta
    mascara_bgr = gerar_mascara(imagem)

    #identificando as bordas das formas geometricas no canto da etiqueta
    bordas = gerar_bordas(mascara_bgr)

    #identificando os contornos da imagem
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centros = []

    for contorno in contornos:
        #aproxima o contorno para uma forma mais simples
        approx = aproxima_contorno(contorno)
        #calcula o centroide do contorno
        centro = calcular_centroide(imagem,contorno)
        #se for um quadrado/retangulo --> transforma ele na origem
        if len(approx) == 4:  
            centro["origem"]=True
            centros.append(centro)
        else:
            centros.append(centro)    

    origem = [c for c in centros if "origem" in c][0]
    outros_pontos = [c for c in centros if "origem" not in c]
    
    #calcula a distancia de cada circulo até o quadrado
    for ponto in outros_pontos:
        ponto["distancia"]=math.dist((ponto["x"],ponto["y"]),(origem['x'],origem['y']))
    
    #ordena os pontos da menor distancia até a maior distancia
    outros_pontos = sorted(outros_pontos, key=lambda d: d['distancia'])

    #cria os pontos de destino e origem
    destino = np.array([[0, 0], [LARGURA_MAXIMA, 0], [LARGURA_MAXIMA, ALTURA_MAXIMA], [0, ALTURA_MAXIMA]], dtype="float32")
    inicio = np.array([[origem['x'],origem['y']],[outros_pontos[1]['x'],outros_pontos[1]['y']],[outros_pontos[2]['x'],outros_pontos[2]['y']],[outros_pontos[0]['x'],outros_pontos[0]['y']]])
    
    #aplica a mudança de perspectiva na etiqueta
    imagem_transformada = transformar_etiqueta(imagem,inicio,destino)

    #retorna uma imagem em escala de cinza
    return cv2.cvtColor(imagem_transformada, cv2.COLOR_BGR2GRAY)

#=========================================================================================================
def preparar_template(template):
    
    # Binariza a imagem do template
    template_binario = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Encontra os contornos na imagem binarizada.
    contornos_template, _ = cv2.findContours(template_binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtra os contornos por área.
    contornos_template_filtrados = [c for c in contornos_template if cv2.contourArea(c) > MIN_AREA]
    
    # Ordena os contornos filtrados da esquerda para a direita.
    contornos_template_filtrados = sorted(contornos_template_filtrados, key=lambda x: cv2.boundingRect(x)[0])
    
    # Dicionário que irá armazenar as imagens recortadas das letras.
    templates = {}
    
    letra_I_template = None
    
    # Verifica se o número de contornos encontrados corresponde ao número de letras esperado.
    if len(contornos_template_filtrados) == len(LETRAS):
        
        # Itera sobre cada contorno ordenado.
        for i, contorno in enumerate(contornos_template_filtrados):
            
            # Pega as coordenadas e dimensões da caixa delimitadora do contorno.
            (x, y, w, h) = cv2.boundingRect(contorno)
            
            # Recorta a região da letra na imagem binarizada.
            letra_template = template_binario[y:y + h, x:x + w]
            
        
            if LETRAS[i] == 'I':
                letra_I_template = letra_template
            else:
                templates[LETRAS[i]] = letra_template
    else:
        # Se o número de contornos não bater, levanta um erro,
        raise ValueError("Template não possui todas as letras")

    templates['I'] = letra_I_template

    # Retorna o dicionário completo com todas as letras recortadas.
    return templates
#===================================================================================================
def analisar_etiqueta(etiqueta, templates):

    _, etiqueta_binaria = cv2.threshold(etiqueta, 127, 255, cv2.THRESH_BINARY_INV)
    contornos_etiqueta, _ = cv2.findContours(etiqueta_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_etiqueta = sorted(contornos_etiqueta, key=lambda c: cv2.boundingRect(c)[1])
    linhas_contornos = agrupar_contornos_por_linha(contornos_etiqueta)
    texto_reconhecido = ""
    img_contornos = cv2.cvtColor(etiqueta, cv2.COLOR_GRAY2BGR)

    for linha in linhas_contornos:
        for contorno in linha:
            x, y, w, h = cv2.boundingRect(contorno)
            area =  cv2.contourArea(contorno)
            if cv2.contourArea(contorno) > MIN_AREA and w <= 200 and h <= 200:
                letra_detectada = etiqueta_binaria[y:y + h, x:x + w]
                cv2.rectangle(img_contornos, (x, y), (x + w, y + h), (255, 0, 0), 1)
                

                #APLICAÇÂO DO MATCHTEMPLATE
                for letra, template in templates.items():
                    letra_detectada_resized = cv2.resize(letra_detectada, (template.shape[1], template.shape[0]))
                    resultado = cv2.matchTemplate(letra_detectada_resized, template, cv2.TM_CCOEFF_NORMED)
                    threshold = 0.8
                    if np.max(resultado) >= threshold:
                        texto_reconhecido += letra
                        cv2.putText(img_contornos, letra, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        break

        texto_reconhecido += '\n'
    #cv2.imshow('Contornos e Letras Detectadas', img_contornos)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return formatar_saida(texto_reconhecido)

#======================================================================================================
imagens = ['im1.png', 'im2.png', 'im3.png', 'im4.png', 'im5.png', 'im6.png', 'im7.png', 'im8.png']


template_letras = cv2.imread(f'banco_de_imagens/template_letras.png', cv2.IMREAD_GRAYSCALE)
template = preparar_template(template_letras)


for imagem in imagens:
    I = cv2.imread(f'banco_de_imagens/{imagem}')
    etiqueta = encontrar_etiqueta(I)
    texto_reconhecido = analisar_etiqueta(etiqueta,template)
    print(f'\nTexto da Imagem {imagem}')
    print(texto_reconhecido)
    print('\n')
