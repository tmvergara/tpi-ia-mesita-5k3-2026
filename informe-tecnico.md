# Trabajo Práctico Integrador - Búsqueda Multimodal de Imágenes

**Materia:** Inteligencia Artificial  
**Universidad:** UTN Facultad Regional Córdoba  
**Curso:** 5K3 Turno Noche
**Cuatrimestre:** Primer Cuatrimestre 2026

**Grupo MesItA**

| Integrante                | Legajo |
| ------------------------- | ------ |
| Ariza Alena               | 95359  |
| Bordino Blanche Juan Cruz | 95008  |
| Vergara Tomas Ignacio     | 94197  |
| Yorlano Pedro             | 95197  |

---

## Índice

- [1. Análisis del Dataset (EDA)](#1-análisis-del-dataset-eda)
  - [1.1 Estadísticas generales](#11-estadísticas-generales)
  - [1.2 Distribución por clases](#12-distribución-por-clases)
  - [1.3 Clustering exploratorio](#13-clustering-exploratorio)
- [2. Metodología](#2-metodología)
  - [2.1 Baseline: CLIP + FAISS](#21-baseline-clip--faiss)
  - [2.2 Pipeline agéntico con Phi-3-mini](#22-pipeline-agéntico-con-phi-3-mini)
  - [2.3 Reranking para consultas con negaciones](#23-reranking-para-consultas-con-negaciones)
  - [2.4 Sobre el ensemble de templates y por qué lo descartamos](#24-sobre-el-ensemble-de-templates-y-por-qué-lo-descartamos)
  - [2.5 Decisiones de diseño - resumen](#25-decisiones-de-diseño--resumen)
- [3. Resultados](#3-resultados)
  - [3.1 Búsqueda baseline](#31-búsqueda-baseline)
  - [3.2 Comparación baseline vs. pipeline agéntico](#32-comparación-baseline-vs-pipeline-agéntico)
  - [3.3 Comparación visual de consultas en español](#33-comparación-visual-de-consultas-en-español)
  - [3.4 Reranking para negaciones](#34-reranking-para-negaciones)
  - [3.5 Ablation study](#35-ablation-study)
- [4. Discusión y Análisis Crítico](#4-discusión-y-análisis-crítico)
- [5. Trabajo Futuro](#5-trabajo-futuro)

---

## 1. Análisis del Dataset (EDA)

### 1.1 Estadísticas generales

El dataset utilizado está basado Pascal VOC 2012 (que consta de imágenes anotadas y es ampliamente utilizado en el contexto de ML y Computer Vision). Para este TPI usamos en realidad un subset específico provisto por la cátedra con imágenes y anotaciones que ya tenemos cargadas como input del Notebook en Kaggle. El análisis se hizo sobre la totalidad del conjunto.

A modo de resumen:

| Estadística                | Valor             |
| -------------------------- | ----------------- |
| Total de imágenes          | 17.125            |
| Clases únicas              | 20                |
| Total de objetos anotados  | 40.138            |
| Ancho promedio             | 466,8 px          |
| Alto promedio              | 389,5 px          |
| Tamaño promedio de archivo | 109,6 KB          |
| Ancho mín / máx            | 142 px / 500 px   |
| Alto mín / máx             | 71 px / 500 px    |
| Tamaño mín / máx           | 7,2 KB / 835,2 KB |

La mayoría de las imágenes tiene ancho fijo de 500 px, lo que indica que el dataset fue normalizado en esa dimensión pero no en el alto. La distribución de tamaños de archivo tiene una cola larga hacia la derecha, con algunas imágenes que superan los 800 KB.

### 1.2 Distribución por clases

Las anotaciones XML fueron parseadas para construir el conteo de objetos por clase y el índice de clase a imágenes. Algo que nos parece valioso de resaltar: una misma imagen puede aparecer en varias clases si contiene múltiples objetos, por lo que la suma de objetos (40.138) supera la cantidad de imágenes (17.125).

| Clase       | Objetos anotados | Imágenes |
| ----------- | ---------------- | -------- |
| person      | 17.401           | 9.583    |
| chair       | 3.056            | 1.366    |
| car         | 2.492            | 1.284    |
| dog         | 1.598            | 1.341    |
| bottle      | 1.561            | 812      |
| cat         | 1.277            | 1.128    |
| bird        | 1.271            | 811      |
| pottedplant | 1.202            | 613      |
| sheep       | 1.084            | 357      |
| boat        | 1.059            | 549      |
| aeroplane   | 1.002            | 716      |
| tvmonitor   | 893              | 645      |
| sofa        | 841              | 742      |
| bicycle     | 837              | 603      |
| horse       | 803              | 526      |
| motorbike   | 801              | 575      |
| diningtable | 800              | 691      |
| cow         | 771              | 340      |
| train       | 704              | 589      |
| bus         | 685              | 467      |

El desbalance de clases es considerable: **person** aparece en más del 55% de las imágenes del dataset, mientras que _cow_ y _sheep_ están presentes en menos de 400. Esto tiene impacto directo en las métricas de evaluación, ya que para la clase **person** hay más de 9.500 imágenes "relevantes", lo que hace que AP@10 sea casi siempre perfecta para esa clase independientemente de la configuración del pipeline.

![distribucion](https://github.com/tmvergara/tpi-ia-mesita-5k3-2026/blob/main/imagenes/eda_distribucion.png?raw=true)

Para entender un poco mejor el dataset, de forma visual que para los humanos es siempre mas sencillo, mostramos una imagen representativa para cada una de estas clases:

![representativas](https://github.com/tmvergara/tpi-ia-mesita-5k3-2026/blob/main/imagenes/eda_imagenes_representativas.png?raw=true)

### 1.3 Clustering exploratorio

Para explorar la estructura visual del dataset antes de construir el índice completo, generamos embeddings CLIP para una muestra de 400 imágenes (semilla aleatoria fija en 42) y aplicamos K-Means con 20 clusters (uno por clase) seguido de reducción de dimensionalidad con UMAP para visualización en 2D.

El resultado muestra que los embeddings de CLIP agrupan imágenes visualmente similares sin ningún entrenamiento específico sobre VOC. Los clusters no coinciden exactamente con las clases semánticas del dataset (una imagen de un auto en una calle puede quedar cerca de imágenes de camiones o buses), pero la separación visual es clara: animales, vehículos, personas y objetos interiores se ubican en zonas distintas del espacio 2D.

![umap](https://github.com/tmvergara/tpi-ia-mesita-5k3-2026/blob/main/imagenes/eda_plot_umap.png?raw=true)

> Plot UMAP (que gráfico bidimensional que se utiliza para visualizar datos de alta dimensionalidad y nos sirve para visualizar como un modelo como CLIP agrupa conceptos, palabras o fotos en su espacio latente.) con puntos coloreados por cluster K-Means (izquierda) y por clase real del dataset (derecha). Esta seccion nos parecio visualmente muy interesante por lo que vale la pena analizarla a detalle ya que revela mucho sobre CLIP.
>
> - En el plot izquierdo: Los embeddings CLIP proyectados con UMAP muestran una estructura semántica clara: clases visualmente similares o relacionadas tienden a agruparse espacialmente. Se pueden identificar algunos patrones concretos:
>   - Las clases de animales (dog, cat, horse, cow, sheep, bird) aparecen en zonas relativamente próximas, lo que refleja que CLIP captura similitud visual/semántica entre categorías relacionadas.
>   - Clases con apariencia muy característica y homogénea (como aeroplane o train) forman clusters más compactos y aislados, lo que sugiere alta separabilidad semántica en el espacio de embeddings.
>   - Clases "genéricas" con mucha variabilidad intraclase (person, chair, sofa) aparecen dispersas o mezcladas con otras, lo cual tiene sentido: una persona puede aparecer en contextos muy distintos.

La superposición entre clase que podemos notar indica por qué las consultas complejas son difíciles. Las clases con alta dispersión intra-categoria (person, bottle) explican cuantitativamente por qué queries del tipo "persona corriendo en campo abierto" son más ambiguas que "avión en vuelo".

---

## 2. Metodología

### 2.1 Baseline: CLIP + FAISS

El baseline consiste en tomar la consulta del usuario tal como la escribió, generar un embedding de texto con CLIP y buscar las imágenes más similares en el índice FAISS.

**Generación de embeddings.** Se usó CLIP ViT-B/32 (probamos otras versiones que degradaban performance, lo exploramos mas adelante). Las imágenes se procesaron en batches de 32 con el preprocesamiento estándar de CLIP (resize, center crop, normalización). Los embeddings resultantes son vectores de 512 dimensiones, normalizados a norma unitaria.

**Índice FAISS.** Se construyó un índice `IndexFlatIP` (producto interno) sobre los 17.125 vectores normalizados. Esto es matemáticamente equivalente a similitud coseno. Se eligió búsqueda exacta (sin aproximación IVF) por dos razones: con 17K vectores de 512 dimensiones el índice ocupa aproximadamente 35 MB en RAM, que es manejable; y la búsqueda exacta evita introducir error adicional que dificulte comparar los resultados entre configuraciones del pipeline.

**Función de búsqueda baseline.** Toma el texto de la consulta, lo tokeniza con `clip.tokenize`, genera el embedding de texto normalizado, y llama a `index.search` para recuperar los top-k más similares.

### 2.2 Pipeline agéntico con Phi-3-mini

La novedad del trabajo es una capa agéntica que preprocesa la consulta antes de ejecutar la búsqueda. Nuestro pipeline tiene cinco módulos con responsabilidades separadas, que se ejecutan en secuencia y registran sus decisiones en una traza (nos parecio vital para entender que sucede, no queriamos trabajar con una caja negra, y ademas para poder mejorar los prompts y reglas en base a los resultados).

#### Por qué un LLM

CLIP fue entrenado principalmente en texto en inglés. Una consulta en español produce embeddings de menor calidad porque el modelo vio proporcionalmente menos contexto en otros idiomes durante el entrenamiento. Además, CLIP no maneja negaciones de forma confiable: el embedding de "car not red" por ejemplo se parece más al de "red car" que al de "car", porque el modelo tiende a capturar los conceptos presentes e ignorar la negación. Para tratar estos casos, es necesaria una etapa previa que entienda la intención semántica de la consulta y esto lo logramos con preprocesamiento usando un LLM relativamente liviano.

#### Por qué Phi-3-mini-4k-instruct

Evaluamos las opciones disponibles en el rango de modelos livianos ejecutables en Kaggle considerando que podiamos usar Aceleramiento por GPU T4 x2, 32 GB RAM y el CPU:

| Modelo                     | Parámetros | RAM GPU (fp16) | Calidad instrucción estructurada                 |
| -------------------------- | ---------- | -------------- | ------------------------------------------------ |
| TinyLlama-1.1B             | 1,1B       | ~1,5 GB        | Baja: output inconsistente en JSON               |
| Llama-3.2-1B               | 1B         | ~1,5 GB        | Media                                            |
| Gemma-2-2B                 | 2B         | ~3 GB          | Media/alta                                       |
| **Phi-3-mini-4k-instruct** | **3,8B**   | **~2,5 GB**    | **Alta: diseñado para instrucción estructurada** |

La elección se justifica en tres puntos: Phi-3-mini fue entrenado por Microsoft con foco explícito en seguir instrucciones con formato exacto (lo que reduce fallas de parseo de JSON), la relación calidad/costo es favorable dado que Kaggle T4 tiene margen amplio con CLIP cargado, y el modelo está disponible directamente en Hugging Face sin dependencias adicionales. TinyLlama fue descartado específicamente porque en pruebas tempranas fallaba con frecuencia al generar los campos JSON que requiere el pipeline, lo que habría complicado el manejo de errores en cada módulo.

También evaluamos ViT-L/14, la variante más grande de CLIP, como alternativa al backbone de imagen. El resultado fue que AP@10 cayó respecto a ViT-B/32. Investigando el motivo, encontramos que ViT-L/14 produce embeddings de 768 dimensiones en lugar de 512, y que el espacio semántico más grande es más sensible a las diferencias de distribución entre el texto y las imágenes de VOC. En la práctica, el modelo más grande no siempre gana cuando el dataset de evaluación es pequeño y el ground truth tiene cierta ambigüedad.

#### Módulos del pipeline

**Módulo 1: Detección de idioma.** Heurística determinista basada en caracteres con tilde, eñe y un vocabulario de palabras funcionales del español (artículos, preposiciones, sustantivos comunes del dataset). No se usa el LLM para esta tarea porque es más rápida y confiable con una heurística simple.

**Módulo 2: Traducción.** Si se detecta español, el LLM traduce la consulta al inglés preservando adjetivos y atributos. El prompt es directo: se le pide al modelo que devuelva únicamente la traducción sin explicaciones.

**Módulo 3: Detección de negaciones.** El LLM analiza la consulta y determina si contiene términos a excluir. Devuelve un JSON con tres campos: `has_negation`, `positive_query` (la parte a buscar) y `negative_terms` (lista de conceptos a penalizar). Este módulo corre **antes** de la verificación de integridad: una consulta como "car not red" es una negación válida, no una contradicción. Si el orden fuera invertido, el módulo de integridad podría clasificarla como incoherente y "corregirla" a "red car", que es exactamente lo opuesto de lo que quiere el usuario.

**Módulo 4: Verificación de integridad.** Finalmente lo diseñamos en dos capas, con el mismo criterio aplicado en el Módulo 1: primero una heurística determinista, luego el LLM. La heurística detecta patrones de contradicción conocidos (pares de formas mutuamente excluyentes como **{square, circle}**, e imposibilidades físicas como **underwater** + **airplane/flying**) sin depender del modelo. Esto fue necesario porque en nuestras pruebas evidenciamos que Phi-3-mini clasifica erróneamente consultas como "square circle in the sky" o "underwater airplane flying" como válidas cuando solo se le proveen ejemplos en el prompt: el modelo captura el contexto general ("escena con elementos visuales") y no razona sobre la contradicción central por la propia naturaleza simple y pequeña del LLM. La capa LLM cubre el resto de los casos no contemplados por la heurística (por ejemplo, conceptos abstractos no representables visualmente). El módulo devuelve **is_valid** y, si es **False**, una **corrected_query** con el término contradictorio removido.

**Módulo 5: Expansión semántica.** El LLM genera 2 variantes de la consulta usando sinónimos o términos relacionados. Las variantes se usan en la búsqueda con pesos: 1.0 para la consulta principal y 0.5 para las variantes adicionales. Los scores se acumulan en un diccionario indexado por imagen y se ordenan al final.

#### Orquestador y trazas

El orquestador combina los cinco módulos y registra una traza completa de decisiones. Cada paso del pipeline anota su entrada, su salida y el motivo de su decisión. Esto permite revisar el comportamiento del sistema ante cualquier consulta sin necesidad de re-ejecutar.

Ejemplo de traza para la consulta "perro corriendo en el parque":

```
[LANGUAGE DETECTION]  result: es
[TRANSLATION]         input: perro corriendo en el parque
                      output: dog running in the park
[NEGATION DETECTION]  has_negation: False
[INTEGRITY CHECK]     is_valid: True
[SEMANTIC EXPANSION]  variants: ['dog running in the park',
                                 'canine sprinting in the green space',
                                 'dog dashing through the park']
[SEARCH]              positive_query_used: dog running in the park
```

Ejemplo de traza para "car not red":

```
[LANGUAGE DETECTION]  result: en
[NEGATION DETECTION]  has_negation: True
                      positive_query: car
                      negative_terms: ['red car']
[INTEGRITY CHECK]     is_valid: True  (evalúa solo "car")
[SEMANTIC EXPANSION]  expanded: False  (no expande cuando hay negación)
[SEARCH]              positive_query_used: car
```

### 2.3 Reranking para consultas con negaciones

Cuando el módulo de negaciones detecta términos a excluir, se aplica un reranking sobre los resultados iniciales. El mecanismo:

1. Se generan embeddings CLIP para cada término negativo.
2. Se calcula la similitud coseno entre ese embedding y el de cada imagen candidata.
3. Se resta al score original una penalización proporcional a esa similitud: `score_reranked = score - penalty_weight * neg_sim`.
4. Se reordenan los resultados por score penalizado.

El parámetro `penalty_weight = 0.5` fue elegido como punto de equilibrio: penaliza suficientemente las imágenes con alta similitud al término negativo sin eliminar imágenes que tengan solo coincidencia incidental. Un peso demasiado alto (probamos con 0.7) haría que imágenes relevantes queden fuera del top-10 por tener el color mencionado en la negación en un objeto secundario de la escena por ejemplo.

### 2.4 Sobre el ensemble de templates y por qué lo descartamos

Durante el desarrollo implementamos también una versión del baseline que promediaba embeddings de texto generados desde múltiples templates del estilo de los usados durante el entrenamiento de CLIP ("a photo of a {}", "a photograph of a {}", etc.). La idea era reducir la varianza del embedding de texto ya que tras leer en internet, los papers originales de OpenAI mostraban incrementos altos en performance al utilizar esta tecnica.

Sin embargo, cuando el pipeline agéntico tiene `should_expand = True`, el orquestador ya está combinando múltiples variantes de la consulta con pesos distintos (1.0 para la variante original, 0.5 para las adicionales). Si además cada variante usara un ensemble de templates internamente, se estarían apilando dos capas de promediado: el promedio de templates dentro de `get_text_embedding_ensemble` y el promedio ponderado entre variantes en el acumulador de scores. El efecto neto es que los scores se "aplanan" hacia valores similares entre sí, lo que degrada la discriminación en el ranking. El denominador del AP@10 se ve afectado porque el ranking relativo entre imágenes se desordena aunque el conjunto recuperado sea parecido al del baseline simple. Decidimos mantener el ensemble comentado en el código y usar embeddings sin template para la función `baseline_search` ya que notamos una degradacion en el score de nuestras submissions al usar el sistema de ensemble de templates.

### 2.5 Decisiones de diseño: resumen

| Decisión                     | Valor / Elección                  | Justificación                                                   |
| ---------------------------- | --------------------------------- | --------------------------------------------------------------- |
| Modelo CLIP                  | ViT-B/32                          | Mejor rendimiento que ViT-L/14 en VOC                           |
| Índice FAISS                 | `IndexFlatIP` (exacto)            | 17K vectores caben en RAM; sin error de aproximación            |
| LLM                          | Phi-3-mini-4k-instruct            | Mejor calidad de instrucción estructurada en el rango liviano   |
| Pesos de expansión semántica | 1.0 (principal) + 0.5 (variantes) | Balance entre recall y precisión en el ranking                  |
| Penalty weight               | 0.5                               | Punto de equilibrio entre penalizar y no sobre-penalizar        |
| Orden de módulos             | Negación antes de integridad      | Evita que negaciones válidas sean tratadas como contradicciones |
| Módulo 4: arquitectura       | Heurística + LLM                  | Phi-3-mini no generaliza contradicciones desde ejemplos solos   |
| Semilla aleatoria            | 42                                | Reproducibilidad en clustering y muestreo                       |

---

## 3. Resultados

### 3.1 Búsqueda baseline

Las pruebas iniciales del baseline con consultas simples en inglés muestran que CLIP recupera imágenes relevantes con scores de similitud coseno entre 0.29 y 0.34 para consultas como "dog", "person on bicycle", "airplane in the sky" y "bus on the street".

![baseline airplane](https://github.com/tmvergara/tpi-ia-mesita-5k3-2026/blob/main/imagenes/baseline_airplane.png?raw=true)
![baseline bus](https://github.com/tmvergara/tpi-ia-mesita-5k3-2026/blob/main/imagenes/baseline_bus.png?raw=true)
![baseline dog](https://github.com/tmvergara/tpi-ia-mesita-5k3-2026/blob/main/imagenes/baseline_dog.png?raw=true)
![baseline person bike](https://github.com/tmvergara/tpi-ia-mesita-5k3-2026/blob/main/imagenes/baseline_person_bike.png?raw=true)

### 3.2 Comparación baseline vs. pipeline agéntico

La comparación se hace sobre las 20 queries con ground truth disponible (q1..q20, clases VOC). La métrica es AP@10.

| qid | Query       | AP Baseline | AP Agéntico | Delta       |
| --- | ----------- | ----------- | ----------- | ----------- |
| q1  | aeroplane   | 1.0000      | 1.0000      | 0.0000      |
| q2  | bicycle     | 1.0000      | 1.0000      | 0.0000      |
| q3  | bird        | 1.0000      | 1.0000      | 0.0000      |
| q4  | boat        | 1.0000      | 1.0000      | 0.0000      |
| q5  | bottle      | 1.0000      | 1.0000      | 0.0000      |
| q6  | bus         | 1.0000      | 1.0000      | 0.0000      |
| q7  | car         | 1.0000      | 1.0000      | 0.0000      |
| q8  | cat         | 1.0000      | 1.0000      | 0.0000      |
| q9  | chair       | 1.0000      | 1.0000      | 0.0000      |
| q10 | cow         | 1.0000      | 1.0000      | 0.0000      |
| q11 | diningtable | 0.8789      | 0.8900      | +0.0111     |
| q12 | dog         | 1.0000      | 1.0000      | 0.0000      |
| q13 | horse       | 1.0000      | 1.0000      | 0.0000      |
| q14 | motorbike   | 0.4509      | 0.6082      | +0.1573     |
| q15 | person      | 0.8900      | 0.8900      | 0.0000      |
| q16 | pottedplant | 0.8789      | 0.8900      | +0.0111     |
| q17 | sheep       | 1.0000      | 1.0000      | 0.0000      |
| q18 | sofa        | 1.0000      | 1.0000      | 0.0000      |
| q19 | train       | 1.0000      | 1.0000      | 0.0000      |
| q20 | tvmonitor   | 1.0000      | 1.0000      | 0.0000      |
|     | **MAP**     | **0.9549**  | **0.9639**  | **+0.0090** |

La mejora más notable es en `motorbike` (+0.1573), donde la expansión semántica generó variantes que mejoran el recall. `diningtable` y `pottedplant` también se benefician levemente. Tenemos que tener en cuenta que solo estamos comparando estas queries basicas por lo que no es esperable tener mejoras sustanciales. Estas se notan cuando la complejidad de la query aumenta como en los q21 a q40, sin embargo para estos casos no teniamos un Ground Truth contra que comparar.

![baseline vs agentico](https://github.com/tmvergara/tpi-ia-mesita-5k3-2026/blob/main/imagenes/ap@10_por_clase_comparacion.png?raw=true)

### 3.3 Comparación visual de consultas en español

Para las consultas "perro corriendo en el parque" y "persona montando bicicleta", se muestran lado a lado los top-5 resultados del baseline (con la query original en español) y del pipeline agéntico (con la query traducida al inglés y expandida semánticamente).

El baseline en español recupera imágenes que tienen baja similitud con la consulta porque CLIP no fue diseñado para texto en español. El pipeline agéntico, al traducir primero, genera un embedding de texto de mayor calidad y recupera imágenes más relevantes.

![car not red](https://github.com/tmvergara/tpi-ia-mesita-5k3-2026/blob/main/imagenes/comparacion_query_agentica_dog.png?raw=true)

![dog not running](https://github.com/tmvergara/tpi-ia-mesita-5k3-2026/blob/main/imagenes/comparacion_query_agentinca_bike.png?raw=true)

> Comparación visual: arriba baseline vs abajo agéntico para las query "perro corriendo en el parque" y "persona montando bicicleta", nos llamó la atención que no se observa una diferencia significativa pero se trata de queries sencillas. Lo que sí observamos es un score mucho mas alto, lo que indica mayor similitud semántica en el espacio de embeddings de CLIP.

### 3.4 Reranking para negaciones

Para las consultas "car not red", "person not happy" y "dog not running" se muestran tres filas: baseline, agéntico sin reranking, y agéntico con reranking.

En "car not red", el pipeline detecta `negative_terms: ['red car']`, busca sobre la parte positiva "car", y el reranking penaliza las imágenes con alta similitud al embedding de "red car". Las imágenes de autos rojos o con elementos rojos prominentes bajan en el ranking, y suben autos de otros colores.

El efecto del reranking no es perfecto: CLIP no tiene una noción explícita de "color de un objeto específico" y mide similitud global de la imagen con el embedding. Autos rojos en escenas complejas pueden quedar con penalización baja si la imagen tiene muchos otros elementos. Pero la tendencia es clara en los ejemplos visuales.

![car not red](https://github.com/tmvergara/tpi-ia-mesita-5k3-2026/blob/main/imagenes/reranking_car_not_red.png?raw=true)

![dog not running](https://github.com/tmvergara/tpi-ia-mesita-5k3-2026/blob/main/imagenes/reranking_dog_not_running.png?raw=true)

> En esta grilla de tres filas (baseline / agéntico sin reranking / agéntico con reranking) para "car not red" y "dog not running", podemos ver visualmente la mejora progresiva que se obtiene al aplicar estas distintas estrategias.

### 3.5 Ablation study

Aprendimos que un Ablation Study es un procedimiento que se utiliza para entender el impacto individual de cada componente de un sistema complejo como nuestro pipeline agentico.
Nos resultó realmente útil ya que al inicio obteníamos deltas negativos, lo que significaba que algunas partes del pipeline estaban literalmente degradando el desempeño. A partir de estas evidencias, pudimos ajustar estas secciones individualmente para mejorar el resultado segun lo que esperabamos.

Para medir el aporte individual de cada componente del pipeline, se definieron cuatro configuraciones acumulativas evaluadas sobre q1..q20:

| Configuración             | Componentes activos                | MAP@10 | Delta   |
| ------------------------- | ---------------------------------- | ------ | ------- |
| A - Baseline puro         | ninguno                            | 0.9549 | nulo    |
| B - + Traducción          | traducción                         | 0.9549 | 0.0000  |
| C - + Expansión semántica | traducción + expansión             | 0.9622 | +0.0073 |
| D - Pipeline completo     | traducción + expansión + reranking | 0.9639 | +0.0017 |

La traducción no aporta sobre q1..q20 porque todas esas queries ya están en inglés. La expansión semántica sube el MAP en 0.0073 puntos, impulsada principalmente por `motorbike`. El reranking agrega un +0.0017 adicional.

![ablation study](https://github.com/tmvergara/tpi-ia-mesita-5k3-2026/blob/main/imagenes/ablation_study.png?raw=true)

> Bar chart del MAP@10 por configuración y heatmap (nos pareció muy evidente y claro) AP@10 por clase y configuración. Acá vemos el impacto principalmente en motorbike o pottedplant.

---

## 4. Discusión y Análisis Crítico

### Qué funcionó bien

**Traducción español → inglés.** El impacto de traducir antes de generar el embedding es visible en los ejemplos cualitativos: las imágenes recuperadas para "perro corriendo en el parque" son claramente mejores con la query traducida. El LLM preserva atributos y complejidad semántica de la consulta original.

**Separación de responsabilidades en el pipeline.** El orden módulo 3 (negaciones) antes de módulo 4 (integridad) resolvió un bug conceptual que encontramos durante el desarrollo inicial: sin ese orden, consultas como "car not red" llegaban al módulo de integridad como algo a "corregir" y el LLM las reformulaba en su opuesto. Con la separación correcta, el módulo de integridad recibe solo "car" y no tiene nada que corregir.

**FAISS con búsqueda exacta.** La decisión de usar `IndexFlatIP` en lugar de aproximaciones IVF fue correcta para este tamaño de dataset. Evitamos introducir error de aproximación y simplificamos el análisis comparativo.

**Trazabilidad.** El sistema de trazas nos permitió entender el comportamiento del pipeline sin necesidad de re-ejecutar. Esto fue útil para debuggear el problema del orden de módulos y para verificar que el LLM generaba los campos JSON esperados en cada caso de prueba.

### Qué no funcionó como esperábamos

**ViT-L/14.** Probamos el backbone más grande de CLIP esperando una mejora en AP@10, pero el resultado fue peor que ViT-B/32. El espacio de embeddings de 768 dimensiones es más sensible a diferencias de distribución entre texto e imágenes, y en un dataset como VOC (con mucho ruido visual y objetos múltiples por imagen) no se traduce en mejor recuperación. El resultado fue contraintuitivo pero consistente entre múltiples pruebas.

**Ensemble de templates.** La idea de promediar embeddings de texto generados desde múltiples templates de entrenamiento de CLIP parecía razonable en teoría. El problema apareció cuando se combinó con la expansión semántica del pipeline agéntico: dos capas de promediado producen scores más uniformes, lo que aplana el ranking y baja AP@10.

**Verificación de integridad (primera versión solo LLM.)** En la versión inicial, el Módulo 4 delegaba toda la detección al LLM. Phi-3-mini clasificaba consultas contradictorias como "square circle in the sky" o "underwater airplane flying" como **is_valid: True** porque capturaba el contexto general de la escena en lugar de razonar sobre la contradicción. La solución fue agregar una capa heurística determinista previa (similar al Módulo 1 de deteccion de idiomas) que detecta patrones de contradicción conocidos, reservando el LLM para casos no contemplados. Esto tiene que ver con distinción más sutil pero intrinsecamente compleja que excede lo que podemos capturar de forma confiable con un modelo pequeño.

**Limitaciones de AP@10 como métrica local.** Las queries q21..q40 no tienen ground truth local disponible, por lo que toda la evaluación cuantitativa se concentra en q1..q20. Estas queries son nombres de clase simples en inglés, para los que CLIP ya funciona muy bien sin ningún preprocesamiento. Esto hace que el ablation study subestime el aporte real del pipeline, ya que los casos donde más ayuda (consultas en español, negaciones, consultas compuestas) no están representados en el ground truth con el que comparamos.

### Limitaciones generales

CLIP mide similitud global texto-imagen y no tiene representación explícita de atributos locales como el color de un objeto específico o la acción de una persona en particular. El reranking por negaciones funciona a nivel de imagen completa, lo que limita su efectividad cuando el elemento a excluir es solo parte de la escena.

El modelo LLM agrega latencia en cada consulta (aproximadamente 1-3 segundos por llamada a Phi-3-mini). En un sistema con muchas consultas concurrentes, esto sería un cuello de botella. Para el contexto del trabajo no es un problema, pero es algo a tener en cuenta si el sistema escalara.

---

## 5. Trabajo Futuro

Con más tiempo y recursos, las mejoras más interesantes a explorar en nuestra opinión serían:

**Evaluación sobre consultas complejas.** Construir un subconjunto anotado manualmente para las queries q21..q40 permitiría medir el impacto del pipeline en los casos para los que fue diseñado. También se podría aproximar el ground truth de consultas compuestas descomponiéndolas en componentes simples y tomando la intersección de resultados parciales.

**Reranking con modelos de visión más capaces.** En lugar de usar CLIP para generar el embedding del término negativo, se podría usar un modelo de detección de objetos o segmentación para identificar si el atributo a excluir aparece en la región principal de la imagen. Esto resolvería el problema de que CLIP penaliza imágenes donde el color/atributo negado aparece en un objeto secundario de la escena.

**Caché de consultas LLM.** Para consultas repetidas o variantes similares, cachear las respuestas del LLM reduciría la latencia sin perder calidad.

**Mejoras en la detección de idioma.** La heurística basada en vocabulario fijo tiene casos límite para consultas mixtas o con nombres propios. Un clasificador de idioma liviano (como `langdetect` o un modelo pequeño de HuggingFace) sería más robusto.

**Explorar CLIP con fine-tuning sobre VOC.** El modelo base de CLIP fue entrenado sobre datos web de propósito general. Un fine-tuning sobre pares texto-imagen extraídos de las anotaciones de VOC podría mejorar el baseline sin necesidad del pipeline agéntico para las consultas simples.
