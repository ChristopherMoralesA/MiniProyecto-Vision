#Importar las librerías por utilizar
import os
import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from skimage.transform import resize

# 8-vecindad
VECINDAD = [
    (-1,-1), (-1, 0), (-1, 1),
    ( 0,-1),          ( 0, 1),
    ( 1,-1), ( 1, 0), ( 1, 1)]

# Maxima desviacion estandar de pixeles no cromaticos
MAX_STD=10

# Minimo porcentaje de pixeles cromaticos
MIN_CROM = 0.1  #10%

# Definicion de semillas para cada agujero
UP     = (120, 70)
DOWN   = (120,250)
RIGHT  = (200,160)
LEFT   = ( 40,160)
CENTER = (120,160)
SEEDS  = (UP,DOWN,RIGHT,LEFT,CENTER)

# Definicion de colores para cada agujero
UP_COLOR     = (255,  0,  0)
DOWN_COLOR   = (255,128,  0)
RIGHT_COLOR  = (  0,204,  0)
LEFT_COLOR   = (  0,128,255)
CENTER_COLOR = (204,  0,102)
SEED_COLORS  = (UP_COLOR,DOWN_COLOR,RIGHT_COLOR,LEFT_COLOR,CENTER_COLOR)

# Minimo de pixeles para considerar ubicacion de agujero
MIN_PX_PERF = 100

# Convierte la imagen RGBA a RGB
def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'
    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    a = np.asarray( a, dtype='float32' ) / 255.0
    R, G, B = background
    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B
    return np.asarray( rgb, dtype='uint8' )

# Toma una imagen y determina si el objeto es cromático
def is_cromatic(image):
    image = rgba2rgb(image)
    rows,columns = image.shape[:2]
    pxl_sum = rows*columns
    clr_pxl = 0
    for row in range(rows):
        for column in range(columns):
            r = image[row][column][0]
            g = image[row][column][1]
            b = image[row][column][2]
            desvest = np.std([r,g,b])
            if desvest > MAX_STD:
                clr_pxl += 1
            if clr_pxl >= MIN_CROM*pxl_sum:
                return True
    return False

# Binariza una imagen (umbral default con método Otsu)
def binarize(image,thresh=0):
    if thresh==0:
        thresh = threshold_otsu(image)
    binary = image > thresh
    return binary

# Crea un región con una semilla y actualiza la imagen out_img
# con la región nueva y su color específico
def region_growth(
    seed,seed_color,bin_img,
    out_img = np.zeros(shape=(320,240,3), dtype=np.uint8)):  
    h, w = bin_img.shape
    region = [seed]
    # Identifica si en la semilla hay una perforacion
    for seed in region:
        x = seed[0]
        y = seed[1]
        if bin_img[y][x] != True:
            return out_img
        out_img[y][x] = seed_color
    # imagen de pixeles visitados, evita revisitar pixeles
    visitado = np.zeros(shape=(bin_img.shape), dtype=np.uint8)
    # Crecimiento de la región
    while len(region):
        seed = region.pop(0)
        x = seed[0]
        y = seed[1]
        # Marca la semilla como visitada
        visitado[y][x] = 1
        for vecino in VECINDAD:
            cur_x = x + vecino[0]
            cur_y = y + vecino[1]
            # limites de la imagen
            if cur_x <0 or cur_y<0 or cur_x >= w or cur_y >=h :
                continue
            # crea nueva semilla si el pixel actual es igual a la semilla
            # y si aun no está visitado, marca pixel actual como visitado
            if (not visitado[cur_y][cur_x]) and (bin_img[cur_y][cur_x]==bin_img[y][x]):
                out_img[cur_y][cur_x] = seed_color
                visitado[cur_y][cur_x] = 1
                region.append((cur_x,cur_y))
    return out_img

# Crea una lista booleana que indica si hay o no agujeros
def agujeros(image):
    ctr = [0,0,0,0,0]
    w, h = image.shape[:2]
    # Conteo de pixeles por agujero
    for i in range(w):
        for k in range(h):
            for j in range(len(ctr)):
                if np.array_equal(image[i][k],SEED_COLORS[j]):
                    ctr[j] += 1
    res_agujeros = [False,False,False,False,False]
    # Evaluar si hay suficientes pixeles para considerarlo agujero
    for j in range(len(ctr)):
        if ctr[j]>=MIN_PX_PERF:
            res_agujeros[j] = True
    return res_agujeros

# Se realiza el crecimiento de regiones para cada agujero y envia resultado
def segmentacion_agujeros(bin_image):
    out_img = np.zeros(shape=(320,240,3), dtype=np.uint8)
    for j in range(len(SEEDS)):
        out_img = region_growth(SEEDS[j],SEED_COLORS[j],bin_image,out_img)
    res_agujeros = agujeros(out_img)
    return res_agujeros

def res_report(res_cromatic,res_agujeros):
    #Resultado cromatico
    cromatic_report = ' es no cromatico y '
    if res_cromatic == True:
            cromatic_report = ' es    cromatico y '
    #Resultado de los agujeros
    agujeros_report = 'hay perforaciones en las posiciones:'
    if np.array_equal(res_agujeros,[False,False,False,False,False]):
        agujeros_report = 'no se encontraron agujeros.'
        report = cromatic_report + agujeros_report
        return report
    if res_agujeros[0]: agujeros_report = agujeros_report + ' Superior'
    else: agujeros_report = agujeros_report + '         '
    if res_agujeros[1]: agujeros_report = agujeros_report + ' Inferior'
    else: agujeros_report = agujeros_report + '         '
    if res_agujeros[2]: agujeros_report = agujeros_report + ' Derecha'
    else: agujeros_report = agujeros_report + '        '
    if res_agujeros[3]: agujeros_report = agujeros_report + ' Izquierda'
    else: agujeros_report = agujeros_report + '          '
    if res_agujeros[4]: agujeros_report = agujeros_report + ' Central'
    report = cromatic_report + agujeros_report
    return report

# Escala y Recorta la imagen a 320x240
def edit_image(image,gray=False):
    if gray == True:
        y,x = image.shape
    else:
        y,x,l = image.shape
    y_div = y//3
    x_div = x//3
    cropped = image[y_div:2*y_div,x_div:2*x_div]
    resized = resize(cropped,(320,240),preserve_range=True).astype(int)
    return resized

# Ejecuta las funciones secuencialmente
def proyecto(image,bin_image):
    image = edit_image(image)
    bin_image = edit_image(bin_image,True)
    res_cromatic = is_cromatic(image)
    res_agujeros = segmentacion_agujeros(bin_image)
    return (res_report(res_cromatic,res_agujeros))

# Toma las imagenes de la carpeta "Objetos_por_analizar"
# Para cada imagen ejecuta el proceso "proyecto"
def main():
    try:
        os.remove("report_file.txt")
    except:
        report_file = open("report_file.txt", "x")
    report_file = open("report_file.txt", "x")
    path = os.getcwd()
    folder = 'Objetos_por_analizar'
    folder_path = os.path.join(path, folder)
    files = os.listdir(folder_path)
    for file in files:
        res_report = ''
        file_path = os.path.join(folder_path, file)
        foto = io.imread(file_path)
        foto_bin= binarize(io.imread(file_path,True))
        res_report =file + proyecto(foto,foto_bin) + '\n'
        report_file.write(res_report)
    report_file.close()

if __name__ == "__main__": main()