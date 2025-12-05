# Predicción con nuevos datos

Este capítulo explica cómo utilizar el modelo entrenado para generar predicciones utilizando `src/prediction.py`.

---

##Ejecutar predicción desde la terminal

Para hacer una predicción, usa:

```bash
python src/prediction.py --input data/raw/example_input.csv --output data/predictions.csv


## ¿Qué hace el script `prediction.py`?

Este script carga el modelo previamente entrenado y genera una predicción para cada fila del archivo CSV que se pase como entrada.  
El resultado se guarda en un archivo de salida, añadiendo una columna llamada `prediction` con los valores 0 o 1 (no sobrevive / sobrevive).


##Ejemplo de archivos

### Archivo de entrada (`example_input.csv`)
Debe contener las mismas columnas que se usaron para entrenar el modelo:


### Archivo de salida (`predictions.csv`)
Después de ejecutar el script, se genera algo como:


---

## Conclusión

Este script permite utilizar el modelo entrenado para generar predicciones con nuevos datos de forma sencilla y reproducible.  
En proyectos reales de MLOps, este sería el componente encargado de realizar inferencias en producción.



