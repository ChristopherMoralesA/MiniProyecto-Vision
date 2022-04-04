#Importar las librerías por utilizar
import os
import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from skimage.transform import resize

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

def is_cromatic(image):
    image = rgba2rgb(image)
    rows,columns,pixel = image.shape
    pxl_sum = rows*columns
    clr_pxl = 0
    for row in range(rows):
        for column in range(columns):
            r = image[row][column][0]
            g = image[row][column][1]
            b = image[row][column][2]
            desvest = np.std([r,g,b])
            if desvest > 10:
                clr_pxl += 1
            if clr_pxl >= 0.1*pxl_sum:
                return True
    return False

def binarize(image,thresh=0):
    if thresh==0:
        thresh = threshold_otsu(image)
    binary = image > thresh
    return binary

def region_growth(seed,seed_color,bin_img,out_img = np.zeros(shape=(320,240,3), dtype=np.uint8)):  
    #out_img = np.zeros(shape=(img.shape), dtype=np.uint8)
    h, w = bin_img.shape
    seeds = [seed]
    #los pixeles de la imagen de saldia que representan las semillas se definen a un valor
    for seed in seeds:
        x = seed[0]
        y = seed[1]
        #En caso de que en la imagen original el pixel que corresponde a la semilla es negro, es decir
        #no es un agujero, no lo pinta, sino que se sale
        if bin_img[y][x] != True:
            return out_img
        out_img[y][x] = seed_color
    #direcciones para a partir de la semilla hacer un cuadrado de la vecindad
    directs = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1),(-1,1),(-1,0)]
    #imagen de pixeles visitados, inicia en cero
    visited = np.zeros(shape=(bin_img.shape), dtype=np.uint8)
    while len(seeds):
        seed = seeds.pop(0)
        x = seed[0]
        y = seed[1]
        # visit point (x,y)
        visited[y][x] = 1
        for direct in directs:
            cur_x = x + direct[0]
            cur_y = y + direct[1]
            # illegal 
            if cur_x <0 or cur_y<0 or cur_x >= w or cur_y >=h :
                continue
            # Not visited and belong to the same target 
            #Checks actual pixel
            if (not visited[cur_y][cur_x]) and (bin_img[cur_y][cur_x]==bin_img[y][x]) :
                out_img[cur_y][cur_x] = seed_color
                visited[cur_y][cur_x] = 1
                seeds.append((cur_x,cur_y))
    return out_img

def agujeros(image, seed_colors):
    up_pxl = 0
    down_pxl = 0
    right_pxl = 0
    left_pxl = 0
    mid_pxl = 0
    res = []
    y, x, l = image.shape
    for i in range(y):
        for k in range(x):
            if np.array_equal(image[i][k],seed_colors['up']):
                up_pxl += 1
            elif np.array_equal(image[i][k],seed_colors['down']):
                down_pxl += 1
            elif np.array_equal(image[i][k],seed_colors['right']):
                right_pxl += 1
            elif np.array_equal(image[i][k],seed_colors['left']):
                left_pxl += 1
            elif np.array_equal(image[i][k],seed_colors['mid']):
                mid_pxl += 1
    res = [up_pxl, down_pxl, right_pxl, left_pxl, mid_pxl]
    res_agujeros = [False,False,False,False,False]
    if res[0]>=100:
        res_agujeros[0] = True
    if res[1]>=100:
        res_agujeros[1] = True
    if res[2]>=100:
        res_agujeros[2] = True
    if res[3]>=100:
        res_agujeros[3] = True
    if res[4]>=100:
        res_agujeros[4] = True
    return res_agujeros

def segmentacion_agujeros(bin_image):
    #Definicion de semillas
    up = (120,90)
    down = (120,240)
    right = (180,150)  
    left = (50,150)
    mid = (125,150)
    #Definicion de colores para cada agujero
    up_color = [255,0,0]
    down_color = [255,128,0]
    right_color = [0,204,0]
    left_color = [0,128,255]
    mid_color = [204,0,102]
    seed_colors = {'up':up_color,'down':down_color,'right':right_color,'left':left_color,'mid':mid_color}
    #Se define la imagend de salida donde se guardaran los agujeros encontrados
    out_img = np.zeros(shape=(320,240,3), dtype=np.uint8)
    #Se realiza el crecimiento de regiones
    out_img = region_growth(up,seed_colors['up'],bin_image,out_img)
    out_img = region_growth(down,seed_colors['down'],bin_image,out_img)
    out_img = region_growth(right,seed_colors['right'],bin_image,out_img)
    out_img = region_growth(left,seed_colors['left'],bin_image,out_img)
    out_img = region_growth(mid,seed_colors['mid'],bin_image,out_img)
    res_agujeros = agujeros(out_img,seed_colors)
    return res_agujeros

def res_report(res_cromatic,res_agujeros):
    #Resultado cromatico
    cromatic_report = ' es no cromatico y '
    if res_cromatic == True:
            cromatic_report = ' es cromatico y '
    #Resultado de los agujeros
    agujeros_report = 'se encontraron agujeros en las siguientes posiciones:'
    if np.array_equal(res_agujeros,[False,False,False,False,False]):
        agujeros_report = 'no se encontraron agujeros.'
        report = cromatic_report + agujeros_report
        return report
    if res_agujeros[0]:
        agujeros_report = agujeros_report + ' -Superior'
    if res_agujeros[1]:
        agujeros_report = agujeros_report + ' -Inferior'
    if res_agujeros[2]:
        agujeros_report = agujeros_report + ' -Lateral derecha'
    if res_agujeros[3]:
        agujeros_report = agujeros_report + ' -Lateral izquierda'
    if res_agujeros[4]:
        agujeros_report = agujeros_report + ' -Central'
    report = cromatic_report + agujeros_report
    return report

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

def proyecto(image,bin_image):
    image = edit_image(image)
    bin_image = edit_image(bin_image,True)
    res_cromatic = is_cromatic(image)
    res_agujeros = segmentacion_agujeros(bin_image)
    return (res_report(res_cromatic,res_agujeros))

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