import os
import string
import time
import numpy as np
from nltk import download
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc

# Descarga de recursos necesarios de NLTK
inicio = time.time()
download('punkt')
download('stopwords')

# Funciones de utilidad
def read_files_in_directory(directory):
    files_contents = []
    file_names = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    files_contents.append(file.read())
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='windows-1252') as file:
                    files_contents.append(file.read())
            file_names.append(filename)
    return file_names, files_contents


def preprocess(text):
    """ Procesa el texto aplicando normalización, eliminación de stopwords y stemming. """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)


def generate_vector_space_models(original_texts, suspicious_texts):
    """ Genera modelos de espacio vectorial para textos originales y sospechosos. """
    if not all(isinstance(text, str) for text in original_texts + suspicious_texts):
        raise TypeError("All inputs must be strings.")
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3))
    # Combina todos los textos para crear un espacio vectorial común
    combined_texts = original_texts + suspicious_texts
    vectorizer.fit(combined_texts)  # Ajusta el vectorizador a todos los textos
    original_vectors = vectorizer.transform(original_texts)  # Transforma los textos originales
    suspicious_vectors = vectorizer.transform(suspicious_texts)  # Transforma los textos sospechosos
    return original_vectors, suspicious_vectors, vectorizer.get_feature_names_out()


def evaluate_performance(similarities, threshold, ground_truth, suspicious_filenames1):
    """
    Evalúa el rendimiento de la herramienta de detección de plagio.
    similarities: matriz de similitudes entre documentos sospechosos y originales.
    threshold: el umbral de similitud para considerar un documento como plagiado.
    ground_truth: dict con clave = nombre del documento sospechoso y valor = bool indicando si es plagiado.
    
    Retorna un dict con las métricas TP, FP, TN, FN.
    """
    TP = FP = TN = FN = 0
    for i, susp_filename in enumerate(suspicious_filenames1):
        is_plagiarized = ground_truth[susp_filename]
        # Considera el documento plagiado si alguna similitud supera el umbral.
        detected_as_plagiarized = any(sim > threshold for sim in similarities[i])
        
        if is_plagiarized and detected_as_plagiarized:
            TP += 1
        elif not is_plagiarized and detected_as_plagiarized:
            FP += 1
        elif is_plagiarized and not detected_as_plagiarized:
            FN += 1
        elif not is_plagiarized and not detected_as_plagiarized:
            TN += 1

    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}


def generate_report(performance_metrics, similarities, ground_truth_labels):
    """
    Genera un informe de rendimiento basado en las métricas dadas.
    performance_metrics: dict con TP, FP, TN, FN.
    similarities: lista de valores de similitud entre documentos.
    ground_truth_labels: lista de etiquetas de verdad fundamental (0 para no plagiado, 1 para plagiado).
    """
    TP = performance_metrics["TP"]
    FP = performance_metrics["FP"]
    TN = performance_metrics["TN"]
    FN = performance_metrics["FN"]
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if TP + FP + TN + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Calcular AUC
    fpr, tpr, thresholds = roc_curve(ground_truth_labels, similarities)
    roc_auc = auc(fpr, tpr)

    report = (
        f"True Positives: {TP}\n"
        f"False Positives: {FP}\n"
        f"True Negatives: {TN}\n"
        f"False Negatives: {FN}\n"
        f"Precision: {precision:.2f}\n"
        f"Recall: {recall:.2f}\n"
        f"Accuracy: {accuracy:.2f}\n"
        f"F1 Score: {f1_score:.2f}\n"
        f"AUC (ROC): {roc_auc:.2f}\n"
    )
    print(report)
    # Puedes añadir código aquí para guardar el informe en un archivo si es necesario.



# En el flujo principal del script:
if __name__ == "__main__":
    # Directorios de los documentos
    path_to_originals = './TextosOriginales'
    path_to_suspicious = './TextosConPlagio'
    
    # Cargar y procesar los documentos
    original_filenames, original_texts = read_files_in_directory(path_to_originals)
    suspicious_filenames, suspicious_texts = read_files_in_directory(path_to_suspicious)

    # Preprocesar todos los textos
    processed_originals = [preprocess(text) for text in original_texts]
    processed_suspicious = [preprocess(text) for text in suspicious_texts]

    # Crear modelos de espacio vectorial común
    original_vectors, suspicious_vectors, feature_names = generate_vector_space_models(processed_originals, processed_suspicious)

    # Calcular similitud de coseno
    similarities = cosine_similarity(suspicious_vectors, original_vectors)

    # Reportar resultados
    threshold = 0  # Umbral de similitud


    # Información de ground truth para evaluación
    ground_truth = {
        'FID-01.txt': True,
        'FID-02.txt': True,
        'FID-03.txt': True,
        'FID-04.txt': True,
        'FID-05.txt': True,
        'FID-06.txt': True,
        'FID-07.txt': True,
        'FID-08.txt': True,
        'FID-09.txt': True,
        'FID-10.txt': True,
        'FID-11.txt': True,
        'FID-12.txt': True,
        'FID-13.txt': True,
        'FID-14.txt': True,
        'FID-15.txt': True,
        'FID-16.txt': False,
        'FID-17.txt': False,
        'FID-18.txt': False,
        'FID-19.txt': True,
        'FID-20.txt': False,
        'FID-21.txt': True,
        'FID-22.txt': True,
        'FID-23.txt': True,
        'FID-24.txt': True,
        'FID-25.txt': True,
        'FID-26.txt': False,
        'FID-27.txt': True,
        'FID-28.txt': True,
        'FID-29.txt': False,
        'FID-30.txt': False,
        'FID-31.txt': False,
    }
    
    # Aplanar las similitudes y preparar las etiquetas de ground truth
    all_similarities = []
    ground_truth_labels = []

    for i, filename in enumerate(suspicious_filenames):
        for j in range(len(original_filenames)):
            all_similarities.append(similarities[i][j])
            # Asumimos que ground_truth usa nombres de archivos sospechosos y marca si son plagiados
            ground_truth_labels.append(1 if ground_truth[filename] else 0)

    for i, filename in enumerate(suspicious_filenames):
        print("\n")
        print(f"Top coincidencias para el archivo sospechoso '{filename}':")
        file_similarities = [(original_filenames[j], similarities[i, j]) for j in range(len(original_filenames))]
        # Ordenar las similitudes y tomar el top 5
        top_5_similarities = sorted(file_similarities, key=lambda x: x[1], reverse=True)[:5]
        for original_file, sim in top_5_similarities:
            if sim > threshold:
                print(f"- {original_file} con una similitud del {sim*100:.2f}%")

    fin = time.time()
    print(f'\n El tiempo de ejecución fue de : {fin-inicio}')
    # Evaluación del rendimiento
    performance_metrics = evaluate_performance(similarities, threshold, ground_truth, suspicious_filenames)
    # Llamada a generate_report
    generate_report(performance_metrics, all_similarities, ground_truth_labels)
