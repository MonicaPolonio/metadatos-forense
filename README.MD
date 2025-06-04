# Herramienta Forense de Análisis y Comparación JPEG

## Descripción

Este proyecto implementa una herramienta forense avanzada para el análisis de imágenes JPEG con fines periciales.
Incluye módulos para:

1. **Análisis forense completo de una imagen JPEG** (metadatos, tablas de cuantificación, estructura binaria, hash SHA256).
2. **Comparación forense entre dos imágenes JPEG**, incluyendo análisis de tablas, metadatos y detección heurística preliminar de doble compresión.
3. **Análisis robusto y avanzado de doble compresión JPEG**, basado en el análisis de coeficientes DCT AC, con estimación preliminar de la calidad de compresión primaria y score forense de probabilidad de doble compresión.

## Metodología y bibliografía utilizada

La herramienta integra y adapta técnicas forenses reconocidas en el ámbito de la detección de manipulaciones y autenticación de imágenes digitales:

* **Kornblum, J. (2008). Using JPEG Quantization Tables to Identify Imagery.**
  Utilizado para la identificación de cámaras y software mediante tablas de cuantificación.

* **Mahdian, B., & Nedbal, R. (2009). JPEG Quantization Tables Forensics: A Statistical Approach.**
  Base estadística para detección de manipulaciones a partir del análisis de tablas de cuantificación.

* **Liu, Q., Sung, A. H., & Qiao, M. (2011). A Method to Detect JPEG-Based Double Compression.**
  Aplicación de análisis de coeficientes DCT y densidades marginales/conjuntas.

* **Pevny, T., & Fridrich, J. (2008). Detection of Double-Compression in JPEG Images for Applications in Steganography.**
  Análisis de histograma de coeficientes AC para detección de recompresión.

* **Taimori, A. et al. (2017). A novel forensic image analysis tool for discovering double JPEG compression clues.**
  Estrategias avanzadas de detección cuantización-unaware y semi-aware.

* **Arenas González, D. M. (2011). Análisis forense de imágenes de móviles mediante el uso de metadatos.**
  Extracción y análisis de estructura binaria y metadatos EXIF/IPTC/XMP.

## Requisitos

* Python 3.8+
* Librerías:

  ```bash
  pip install exiftool jpeg-qtables opencv-python numpy
  ```
* Tener instalado `exiftool` en el sistema operativo.

## Uso

Ejecutar el script en consola:

```bash
python forense_jpeg.py
```

### Menú interactivo

1. **Análisis forense completo de una imagen**

   * Extrae y analiza metadatos, tablas de cuantificación, estructura binaria y calcula el hash SHA256.
   * Genera un informe forense en `.txt`.

2. **Comparación forense entre dos imágenes**

   * Compara tablas de cuantificación.
   * Compara metadatos clave.
   * Detecta heurísticamente indicios de doble compresión.
   * Genera informe comparativo en `.txt`.

3. **Análisis avanzado de doble compresión JPEG**

   * Extrae coeficientes DCT.
   * Calcula histograma AC (1,1).
   * Aplica heurística de Pevny y Liu para estimar irregularidades y score de doble compresión.

### Ejemplos de ejecución

* Opcion 1:

  ```bash
  Ruta de la imagen: ejemplo.jpg
  ```

  Resultado: `ejemplo_reporte_forense.txt`

* Opcion 2:

  ```bash
  Ruta de la imagen 1: original.jpg
  Ruta de la imagen 2: sospechosa.jpg
  ```

  Resultado: `reporte_comparativo.txt`

* Opcion 3:

  ```bash
  Ruta de la imagen: sospechosa.jpg
  ```

  Resultado: Muestra en pantalla el score estimado.

## Notas forenses

* **La detección de doble compresión es probabilística.**
* **El score de doble compresión debe considerarse como un indicador preliminar y no una evidencia concluyente.**
* **Se recomienda complementar el análisis con inspección pericial cualificada.**

## Licencia

MIT License. Uso libre con reconocimiento de autoría.

## Autor

