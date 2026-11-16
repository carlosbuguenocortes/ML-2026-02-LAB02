"""Carga de LFW. La selección de identidades la completa el alumno."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_lfw_people

from src.config import DIR_CACHE
from src.datos.base import CargadorRostros, validar_conjunto
from src.modelos import ConjuntoRostros


class CargadorLFW(CargadorRostros):
    """Fotografías de LFW con mayor diversidad de pose, luz y fondo."""

    nombre = "lfw"

    def cargar(self) -> ConjuntoRostros:
        # 1. Descargar y cargar identidades usando 20 fotos con mitad resolucion
        rostros = fetch_lfw_people(
            min_faces_per_person=20,
            resize=0.5,
            data_home=str(DIR_CACHE),
            download_if_missing=True,
        )
        images = np.asarray(rostros.images)
        X = np.asarray(rostros.data, dtype=float)
        y = np.asarray(rostros.target, dtype=int)
        target_names = np.asarray(rostros.target_names, dtype=str)

        # 2. Conservar como maximo las 50 identidades que tienen mas frecuencia
        clases_unicas, conteos = np.unique(y, return_counts=True)
        if clases_unicas.size > 50:
            indices_top = np.argsort(-conteos, kind="stable")[:50]
            clases_conservadas = clases_unicas[indices_top]
            mascara = np.isin(y, clases_conservadas)
            images = images[mascara]
            X = X[mascara]
            y = y[mascara]

        # 3. Reindexar de forma contigua y alinear los nombres de identidad
        clases_finales, y_reindexado = np.unique(y, return_inverse=True)
        y_reindexado = np.asarray(y_reindexado, dtype=int).ravel()
        nombres = [str(target_names[idx]) for idx in clases_finales]

        # 4. Leer alto y ancho con images.shape
        alto, ancho = int(images.shape[1]), int(images.shape[2])

        # 5. Construir la estructura, validar y retornar
        conjunto = ConjuntoRostros(
            nombre=self.nombre,
            images=images,
            X=X,
            y=y_reindexado,
            nombres_identidades=nombres,
            alto=alto,
            ancho=ancho,
        )
        validar_conjunto(conjunto)
        return conjunto