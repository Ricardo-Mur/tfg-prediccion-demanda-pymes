"""Respuestas preparadas para el programa congelado, cada una unida a la pregunta a la que contesta.

El programa pregunta por teclado. La interfaz no teclea: prepara un guion con las respuestas que el usuario ha
dado en pantalla, y el lector de este módulo las entrega una a una, comprobando antes que la pregunta que hace
el programa es la que esa respuesta espera.

Reglas:
  - Las respuestas se usan en orden y una sola vez.
  - Cada paso dice qué debe contener la pregunta y, si la pregunta no se distingue sola, qué debe haber escrito
    antes el programa (por ejemplo, «Grupo 2 de 3»).
  - Si la pregunta no es la esperada, si no queda respuesta, si el programa rechaza una respuesta y vuelve a
    preguntar, o si al terminar sobran respuestas, la ejecución se detiene. Nunca se contesta a ciegas.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass


def reemplazar(temporal, ruta, intentos: int = 100) -> None:
    """`os.replace`, que en Windows falla si otro proceso tiene abierto el archivo justo en ese momento (por ejemplo,
    la interfaz leyendo el estado). Se reintenta durante unos segundos; en Linux no hace falta y sale a la primera."""
    for _ in range(intentos - 1):
        try:
            os.replace(temporal, ruta)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(temporal, ruta)


class DetenerEjecucion(BaseException):
    """La ejecución no puede seguir con garantías.

    Hereda de BaseException, como una interrupción de teclado, para que ningún `except Exception` del programa
    la absorba y siga adelante.
    """

    def __init__(self, motivo: str, detalle: str = ''):
        super().__init__(f'{motivo}: {detalle}' if detalle else motivo)
        self.motivo, self.detalle = motivo, detalle


@dataclass(frozen=True)
class Paso:
    pregunta: tuple[str, ...]          # trozos que debe contener la pregunta
    respuesta: str
    contexto: tuple[str, ...] = ()     # trozos que el programa debe haber escrito desde la pregunta anterior

    def __post_init__(self):
        if not self.pregunta or any(len(t.strip()) < 3 for t in (*self.pregunta, *self.contexto)):
            raise ValueError('Cada respuesta debe ir unida a una pregunta reconocible.')
        if '\n' in self.respuesta or '\r' in self.respuesta:
            raise ValueError('Una respuesta ocupa una sola línea.')

    def a_dict(self) -> dict:
        return {'pregunta': list(self.pregunta), 'respuesta': self.respuesta, 'contexto': list(self.contexto)}

    @staticmethod
    def desde_dict(d: dict) -> 'Paso':
        return Paso(tuple(d['pregunta']), str(d['respuesta']), tuple(d.get('contexto', ())))


def paso(pregunta, respuesta: str, contexto=()) -> Paso:
    """Atajo: admite un texto o varios."""
    como_tupla = lambda x: (x,) if isinstance(x, str) else tuple(x)
    return Paso(como_tupla(pregunta), respuesta, como_tupla(contexto))


# Lo que escribe el programa cuando no acepta una respuesta y vuelve a preguntar, o cuando la ignora.
RECHAZOS = ('No he entendido la respuesta', 'Responde con una de las letras indicadas', 'Escribe un número',
            'Escribe un mes entre', 'Puedes responder S, C o N', 'Elige números de la lista',
            'no está en este grupo', 'no tiene dato en esa columna', 'Esa columna ya corresponde a otro dato',
            'No puedo usarla')


class Lector:
    """Sustituye al teclado. `leer_pantalla()` devuelve lo que el programa ha escrito desde la pregunta anterior."""

    def __init__(self, pasos, leer_pantalla):
        self.pasos, self.leer_pantalla = list(pasos), leer_pantalla
        self.usadas = 0

    def __call__(self, mensaje: str = '') -> str:
        escrito = self.leer_pantalla()
        if self.usadas:
            for rechazo in RECHAZOS:
                if rechazo in escrito:
                    raise DetenerEjecucion('respuesta_rechazada',
                                           f'el programa no ha aceptado la respuesta n.º {self.usadas} («{rechazo}»)')
        if self.usadas >= len(self.pasos):
            raise DetenerEjecucion('pregunta_sin_respuesta', str(mensaje).strip())
        esperado = self.pasos[self.usadas]
        if not all(t in mensaje for t in esperado.pregunta) or not all(t in escrito for t in esperado.contexto):
            raise DetenerEjecucion('pregunta_inesperada',
                                   f'el programa pregunta «{str(mensaje).strip()}» y la respuesta n.º {self.usadas + 1} '
                                   f'espera «{" … ".join(esperado.pregunta)}»'
                                   + (f' después de «{" … ".join(esperado.contexto)}»' if esperado.contexto else ''))
        self.usadas += 1
        return esperado.respuesta

    def comprobar_final(self) -> None:
        """Al terminar el programa no puede sobrar ninguna respuesta."""
        if self.usadas != len(self.pasos):
            raise DetenerEjecucion('respuestas_sin_usar',
                                   f'el programa ha hecho {self.usadas} preguntas y había {len(self.pasos)} respuestas')


class LectorDePreguntas(Lector):
    """Modo de preguntas: primero entrega, comprobándolas, las respuestas ya dadas; después, cada pregunta nueva
    del programa se deja escrita para que la interfaz la enseñe, y se espera la respuesta del usuario.

    Cada respuesta nueva queda grabada con el texto exacto de su pregunta y con todo lo que el programa había
    escrito antes: así el guion del cálculo sale del propio programa, no de una reconstrucción.
    `preguntar(n, mensaje, escrito)` devuelve la respuesta del usuario; `grabar(pasos)` guarda lo grabado.
    """

    def __init__(self, pasos, leer_pantalla, preguntar, grabar):
        super().__init__(pasos, leer_pantalla)
        self.preguntar, self.grabar = preguntar, grabar

    def __call__(self, mensaje: str = '') -> str:
        if self.usadas < len(self.pasos):
            return super().__call__(mensaje)
        escrito = self.leer_pantalla()
        if self.usadas:
            for linea in escrito.splitlines():
                if any(r in linea for r in RECHAZOS):
                    raise DetenerEjecucion('respuesta_rechazada', linea.strip())
        respuesta = str(self.preguntar(self.usadas + 1, mensaje, escrito))
        contexto = tuple(l.strip() for l in escrito.splitlines() if len(l.strip()) >= 3)
        self.pasos.append(Paso((str(mensaje).strip(),), respuesta, contexto))
        self.usadas += 1
        self.grabar(self.pasos)
        return respuesta
