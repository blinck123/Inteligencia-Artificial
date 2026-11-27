# Proyecto CD — detección de tuberculosis en radiografías de tórax
Prototipo académico, no herramienta de diagnóstico clínico.

## Dataset recomendado
- **Prueba rápida del código:** TB Chest X-ray Database (Kaggle / [IEEE DataPort](https://ieee-dataport.org/documents/tuberculosis-tb-chest-x-ray-database), CC BY 4.0). Descarga inmediata con cuenta de Kaggle. Carpetas `Normal` y `Tuberculosis`; sin ID de paciente (se asume 1 imagen = 1 paciente) y compilado de varias fuentes, por lo que pueden existir duplicados.
- **Trabajo/paper:** Shenzhen (662 imágenes: 326 normales, 336 TB) como conjunto principal y Montgomery (138: 80 normales, 58 TB) como validación externa. Se solicitan al NLM (http://archive.nlm.nih.gov/repos/chestImages.php); sin licencia formal, no redistribuir y citar a Jaeger et al. (2014) y Candemir et al. (2014).

## Ejecutar en Windows
1. Abrir esta carpeta en VS Code.
2. `python -m venv .venv`
3. `.\.venv\Scripts\Activate.ps1`
4. `python -m pip install -r requirements.txt`
5. `python preparar_tb.py --origen RUTA_DESCARGA --lado 512` (copia redimensionada a dataset/imagenes y crea anotaciones.csv desde el nombre de archivo o la carpeta; `--max-por-clase 200` para pruebas rápidas en CPU)
6. `python auditar_dataset.py` (balance, duplicados, casi-duplicados, calidad)
7. `python main.py --epochs 10 --batch-size 8 --pretrained`
8. `python validacion_cruzada.py --folds 5 --epochs 10 --batch-size 8 --pretrained`
9. Opcional, solo con anotaciones reales de expertos: `python concordancia.py`

Sin --pretrained se inicia con pesos aleatorios. Con la opción se descargan pesos ImageNet si no están almacenados. Las radiografías en gris se convierten a 3 canales para ResNet18.

## Datos
anotaciones.csv: imagen, paciente_id (seudonimizado), diagnostico (0 normal / 1 tuberculosis). Las columnas cavitacion, infiltrado, derrame_pleural, nodulos (ausente/presente) y zona_afectada (superior/media/inferior/difusa) son una **rúbrica radiológica opcional** (DIM_OPCIONAL en main.py, editable): solo si hay anotaciones reales de expertos y están completas se entrena además el modelo multitarea; si están vacías solo se entrena el modelo base. Los datasets públicos no las incluyen: no inventar etiquetas. Se necesitan suficientes pacientes independientes de ambas clases para las particiones. No usar imágenes sin autorización/licencia. evaluadores.csv: una fila por imagen, evaluador y condición, con las mismas categorías.

## Resultados
Particiones por paciente (~70/15/15), pesos del mejor modelo según AUC de validación, historial, métricas, predicciones de prueba, curvas ROC y matrices de confusión. Pérdida multitarea = CE diagnóstico + 0.5 * promedio CE auxiliares. Kappa de Cohen nominal para la rúbrica; ICC solo si se definen dimensiones ordinales (ORDINALES).

## Validación cruzada
`validacion_cruzada.py` usa StratifiedGroupKFold por paciente (admite pacientes con diagnósticos mixtos); ~1/6 de los pacientes de entrenamiento elige la mejor época. Salidas en `resultados/cv/`: métricas por fold, resumen (media, desv. est., IC95% t), predicciones out-of-fold y, si hay modelo multitarea, comparación pareada (t, Wilcoxon, bootstrap por paciente). Sin rúbrica solo se evalúa el modelo base con IC95% bootstrap del AUC.

## Alcance pendiente
No se ha ejecutado entrenamiento real con datos reales. Si se combinan datasets (p. ej. Shenzhen + Montgomery) hay riesgo de sesgo por fuente: reportar resultados por separado y usar Montgomery como validación externa. Pendiente validar explicaciones (p. ej. Grad-CAM) con expertos; un clasificador no demuestra razonamiento clínico. No usar los resultados para decisiones clínicas.
