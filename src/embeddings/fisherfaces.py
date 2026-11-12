"""Fisherfaces: PCA comprime y LDA discrimina. Implementado por el alumno."""

from __future__ import annotations

import numpy as np

from src.embeddings.base import ExtractorEmbedding
from src.embeddings.pca import ExtractorPCA
from src.embeddings.lda import ExtractorLDA
from src.excepciones import EtapaPendienteAlumno


class ExtractorFisherfaces(ExtractorEmbedding):
    """Proyección z = W_LDAᵀ W_PCAᵀ (x − μ), ajustada solo con TRAIN.

    Orden obligatorio:
    1. Ajustar PCA con ``d_pca`` solo sobre ``X_train`` (ahí se estima μ).
    2. Ajustar LDA con ``d_lda`` sobre ese PCA de TRAIN, usando ``y_train``.
    3. Transformar TEST con las dos proyecciones, sin volver a ajustar.

    Contrato: ``d_lda <= min(C - 1, d_pca)``. 
    """

    nombre = "fisherfaces"

    def __init__(self, d_pca: int, d_lda: int) -> None:
        if d_pca < 1 or d_lda < 1:
            raise ValueError("Fisherfaces requiere d_pca y d_lda positivos.")
        if d_lda > d_pca:
            raise ValueError(
                f"d_lda={d_lda} supera d_pca={d_pca}. "
                "Debe cumplirse d_lda <= min(C-1, d_pca)."
            )
        self.d_pca = int(d_pca)
        self.d_lda = int(d_lda)
        
        # Atributos internos para los extractores
        self.pca: ExtractorPCA | None = None
        self.lda: ExtractorLDA | None = None

    @property
    def dimension(self) -> int:
        return self.d_lda

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ExtractorFisherfaces:
        """Ajusta la tubería secuencial PCA -> LDA sin fuga de información."""
        if y is None:
            raise ValueError("Fisherfaces es un método supervisado y requiere etiquetas 'y'.")

        # 1. Ajustar PCA con d_pca solo sobre el conjunto TRAIN
        self.pca = ExtractorPCA(n_components=self.d_pca)
        self.pca.fit(X)
        X_pca = self.pca.transform(X)

        # 2. Validar la restricción dimensional de LDA: d_lda <= min(C - 1, d_pca)
        clases_unicas = np.unique(y)
        C = len(clases_unicas)
        limite_lda = min(C - 1, self.d_pca)

        if self.d_lda > limite_lda:
            raise ValueError(
                f"d_lda={d_lda} excede el límite permitido min(C-1, d_pca) = {limite_lda} (C={C})"
            )

        # 3. Ajustar LDA con d_lda sobre el espacio PCA de TRAIN usando y_train
        self.lda = ExtractorLDA(n_components=self.d_lda)
        self.lda.fit(X_pca, y)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Proyecta los datos aplicando PCA y luego LDA en secuencia, sin reajustar."""
        if self.pca is None or self.lda is None:
            raise RuntimeError("El extractor Fisherfaces no ha sido ajustado (ejecute fit primero).")

        # Transforma primero por PCA y el resultado pásalo por LDA
        X_pca = self.pca.transform(X)
        X_lda = self.lda.transform(X_pca)
        
        return X_lda