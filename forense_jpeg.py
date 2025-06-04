import exiftool
import jpeg_qtables
import hashlib
import cv2
import numpy as np
from pathlib import Path

# ------------------ MÓDULO 1: EXTRACCIÓN FORENSE ------------------ #
def extraer_metadatos(ruta_imagen):
    with exiftool.ExifTool() as et:
        metadata = et.get_metadata(ruta_imagen)
    return metadata

def formatear_tabla_8x8(tabla):
    return [" | ".join(f"{tabla[i + j]:02X}" for j in range(8)) for i in range(0, 64, 8)]

def extraer_tablas_cuantificacion(ruta_imagen):
    tablas = jpeg_qtables.get_quant_tables(ruta_imagen)
    return [tabla for tabla in tablas]

def calcular_hash(ruta_imagen):
    sha256 = hashlib.sha256()
    with open(ruta_imagen, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def extraer_estructura_jpeg(ruta_imagen):
    bloques = []
    with open(ruta_imagen, 'rb') as f:
        datos = f.read()
    i = 0
    while i < len(datos):
        if datos[i] == 0xFF:
            marcador = datos[i:i+2]
            longitud = int.from_bytes(datos[i+2:i+4], byteorder='big') if i+4 <= len(datos) else 0
            bloques.append({
                'posicion': i,
                'marcador': marcador.hex().upper(),
                'longitud': longitud,
                'contenido_hex': datos[i+4:i+4+16].hex().upper()
            })
            if marcador == b'\xFF\xD9':
                break
            i += 2 + longitud if longitud > 0 else 2
        else:
            i += 1
    return bloques

def generar_reporte_forense(imagen, salida_reporte):
    metadatos = extraer_metadatos(imagen)
    tablas = extraer_tablas_cuantificacion(imagen)
    hash_sha256 = calcular_hash(imagen)
    estructura = extraer_estructura_jpeg(imagen)

    with open(salida_reporte, 'w', encoding='utf-8') as reporte:
        reporte.write(f"ANÁLISIS FORENSE COMPLETO\nImagen: {imagen}\nHash SHA256: {hash_sha256}\n\n")
        reporte.write("--- Metadatos ---\n")
        for clave, valor in metadatos.items():
            reporte.write(f"{clave}: {valor}\n")

        reporte.write("\n--- Tablas de cuantificación (8x8 HEX) ---\n")
        for i, tabla in enumerate(tablas):
            reporte.write(f"\nTabla {i}:\n")
            for fila in formatear_tabla_8x8(tabla):
                reporte.write(f"| {fila} |\n")

        reporte.write("\n--- Estructura binaria JPEG ---\n")
        for bloque in estructura:
            reporte.write(f"Posición: {bloque['posicion']}, Marcador: {bloque['marcador']}, Longitud: {bloque['longitud']}, HEX: {bloque['contenido_hex']}\n")

    print(f"\nInforme forense generado: {salida_reporte}")

# ------------------ MÓDULO 2: COMPARACIÓN FORENSE ------------------ #
def comparar_tablas_cuantificacion(tablas1, tablas2):
    resultados = []
    for i, (tabla1, tabla2) in enumerate(zip(tablas1, tablas2)):
        diferencias = [abs(a - b) for a, b in zip(tabla1, tabla2)]
        media_diferencias = sum(diferencias) / len(diferencias)
        iguales = tabla1 == tabla2
        resultados.append({
            'tabla_id': i,
            'iguales': iguales,
            'media_diferencias': media_diferencias
        })
    return resultados

def comparar_metadatos(metadatos1, metadatos2, claves_importantes):
    diferencias = {}
    for clave in claves_importantes:
        valor1 = metadatos1.get(clave, "N/D")
        valor2 = metadatos2.get(clave, "N/D")
        diferencias[clave] = {
            'imagen1': valor1,
            'imagen2': valor2,
            'iguales': valor1 == valor2
        }
    return diferencias

def analisis_doble_compresion_heuristico(tabla_cuant1, tabla_cuant2):
    posibles = []
    for i, (tabla1, tabla2) in enumerate(zip(tabla_cuant1, tabla_cuant2)):
        if any(t2 >= t1 for t1, t2 in zip(tabla1, tabla2)):
            posibles.append(i)
    return posibles

def generar_reporte_comparativo(imagen1, imagen2, salida_reporte):
    metadatos1 = extraer_metadatos(imagen1)
    tablas1 = extraer_tablas_cuantificacion(imagen1)
    hash1 = calcular_hash(imagen1)

    metadatos2 = extraer_metadatos(imagen2)
    tablas2 = extraer_tablas_cuantificacion(imagen2)
    hash2 = calcular_hash(imagen2)

    tablas_result = comparar_tablas_cuantificacion(tablas1, tablas2)
    metadatos_result = comparar_metadatos(metadatos1, metadatos2, ['EXIF:Make', 'EXIF:Model', 'EXIF:SerialNumber', 'EXIF:Software'])
    sospecha_doble = analisis_doble_compresion_heuristico(tablas1, tablas2)

    with open(salida_reporte, 'w', encoding='utf-8') as reporte:
        reporte.write(f"COMPARACIÓN FORENSE DE IMÁGENES\n\nImagen 1: {imagen1}\nHash: {hash1}\nImagen 2: {imagen2}\nHash: {hash2}\n\n")
        reporte.write("--- Comparación de Metadatos ---\n")
        for clave, resultado in metadatos_result.items():
            reporte.write(f"{clave} - Imagen 1: {resultado['imagen1']} | Imagen 2: {resultado['imagen2']} | Iguales: {resultado['iguales']}\n")

        reporte.write("\n--- Comparación de Tablas de Cuantificación ---\n")
        for res in tablas_result:
            reporte.write(f"Tabla {res['tabla_id']} - Iguales: {res['iguales']} | Media de diferencias: {res['media_diferencias']:.2f}\n")

        reporte.write("\n--- Detección heurística preliminar de doble compresión ---\n")
        reporte.write(f"{'POSIBLE' if sospecha_doble else 'NO'} doble compresión detectada en tablas: {sospecha_doble}\n")

    print(f"\nInforme comparativo generado: {salida_reporte}")


# ------------------ MÓDULO 3 MEJORADO: ANÁLISIS DCT ------------------ #
def cargar_imagen_dct(ruta_imagen):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo cargar la imagen.")
    h, w = img.shape
    h = h - (h % 8)
    w = w - (w % 8)
    img = img[:h, :w]
    bloques = [cv2.dct(np.float32(img[i:i+8, j:j+8]) - 128)
               for i in range(0, h, 8) for j in range(0, w, 8)]
    return bloques

def extraer_histograma_dct_ac(bloques, modo=(1,1)):
    coeficientes = [bloque[modo] for bloque in bloques]
    histograma, _ = np.histogram(coeficientes, bins=range(-50, 51), density=True)
    return histograma

def calcular_metricas_doble_compresion(histograma):
    varianza = np.var(histograma)
    smoothness = np.sum(np.abs(np.diff(histograma)))
    score = min(1.0, (smoothness / (varianza + 1e-5)) * 0.1)
    return varianza, smoothness, score

def analizar_doble_compresion_avanzado(imagen):
    bloques_dct = cargar_imagen_dct(imagen)
    histograma = extraer_histograma_dct_ac(bloques_dct)
    varianza, smoothness, score = calcular_metricas_doble_compresion(histograma)
    return varianza, smoothness, score



# ------------------ MENÚ INTERACTIVO ------------------ #
def menu():
    print("\n--- MENÚ FORENSE ---")
    print("1. Análisis forense completo de una imagen")
    print("2. Comparación forense entre dos imágenes")
    print("3. Análisis avanzado de doble compresión JPEG")
    opcion = input("Selecciona una opción (1-3): ")

    if opcion == "1":
        imagen = input("Ruta de la imagen: ")
        salida = Path(imagen).stem + "_reporte_forense.txt"
        generar_reporte_forense(imagen, salida)

    elif opcion == "2":
        imagen1 = input("Ruta de la imagen 1: ")
        imagen2 = input("Ruta de la imagen 2: ")
        salida = "reporte_comparativo.txt"
        generar_reporte_comparativo(imagen1, imagen2, salida)

    elif opcion == "3":
        imagen = input("Ruta de la imagen: ")
        varianza, smoothness, score = analizar_doble_compresion_avanzado(imagen)
        print(f"\nResultados del análisis avanzado:")
        print(f"Varianza del histograma AC: {varianza:.6f}")
        print(f"Irregularidad (smoothness): {smoothness:.6f}")
        print(f"Score de doble compresión (0 a 1): {score:.2f}")

    else:
        print("Opción no válida.")

# Ejecutar menú
menu()
