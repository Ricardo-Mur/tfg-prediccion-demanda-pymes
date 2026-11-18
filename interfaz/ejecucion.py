"""Lanzar, vigilar y cancelar una ejecución del programa congelado en su espacio.

  - Una sola ejecución a la vez por espacio: un cerrojo impide lanzar dos.
  - La misma petición (mismos archivos, mismas respuestas, mismas opciones) no se calcula dos veces.
  - Cancelar termina el proceso y los procesos de cálculo que haya abierto, y borra lo que hubiera escrito.
  - Solo hay resultados si la ejecución terminó bien. En cualquier otro caso, la carpeta de salidas se borra.
  - Si quien lanzó el cálculo desaparece (se cierra la aplicación), el cálculo se detiene solo.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .espacio import CERROJO, Espacio, huella
from .guion import reemplazar

LANZADOR = Path(__file__).resolve().parent / 'lanzador.py'
FINALES = ('terminada', 'preguntas_completas', 'detenida', 'error', 'sin_resultados', 'cancelada')
REGISTROS = ('estado.json', 'peticion.json', 'pantalla.txt', 'errores.txt', 'error_tecnico.txt', 'intentos_de_red.txt',
             'ultimo_proceso.txt', 'pregunta.json', 'grabado.json')
_PROCESOS: dict[str, subprocess.Popen] = {}     # el servidor de la interfaz vive más que cada pantalla
_ACTIVIDAD: dict[str, dict] = {}                # cuándo se notó por última vez que cada cálculo hacía algo
NULA = {'stdin': subprocess.DEVNULL}            # las órdenes de Windows no heredan la entrada de la aplicación
DIAGNOSTICO, PILAS = 'diagnostico.txt', 'pilas.txt'
VIGILADOS = ('pantalla.txt', 'errores.txt', DIAGNOSTICO, 'pregunta.json', 'grabado.json', 'estado.json')


def registrar(espacio: Path, texto: str) -> None:
    """Una línea de la aplicación en `diagnostico.txt`, junto a las del proceso de cálculo. Nunca falla."""
    try:
        with open(espacio / DIAGNOSTICO, 'a', encoding='utf-8') as f:
            f.write(f'{time.strftime("%H:%M:%S")}            [interfaz {os.getpid()}] {texto}\n')
    except OSError:
        pass


def _cpu_del_arbol(pid: int) -> float | None:
    """Segundos de procesador que llevan gastados el proceso y todos los que ha abierto. None si no se puede saber
    (falta psutil o el proceso ya no existe). En Windows, con un entorno virtual, el proceso lanzado es un pequeño
    lanzador de Python y el intérprete es su hijo: por eso se cuentan los descendientes."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        raiz = psutil.Process(pid)
        procesos = [raiz] + raiz.children(recursive=True)
    except psutil.Error:
        return None
    total = 0.0
    for proceso in procesos:
        try:
            tiempos = proceso.cpu_times()
            total += tiempos.user + tiempos.system
        except psutil.Error:
            pass
    return total


def _pid_vivo(pid: int) -> bool:
    if os.name == 'nt':
        r = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'], capture_output=True, text=True, **NULA)
        return str(pid) in r.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _grupo_vivo(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


class Ejecucion:
    def __init__(self, espacio: Espacio):
        self.espacio = espacio
        self.cerrojo = espacio.ruta / CERROJO
        self.clave = str(espacio.ruta)

    # ------------------------------------------------------------------ estado

    def estado(self) -> dict:
        try:
            return json.loads((self.espacio.ruta / 'estado.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}

    def _anotar(self, **datos) -> None:
        ruta = self.espacio.ruta / 'estado.json'
        temporal = ruta.with_suffix('.tmp')
        temporal.write_text(json.dumps(self.estado() | datos, ensure_ascii=False, indent=1), encoding='utf-8')
        reemplazar(temporal, ruta)

    def _pid(self) -> int | None:
        try:
            return int(self.cerrojo.read_text(encoding='utf-8').strip())
        except (OSError, ValueError):
            return None

    def _vivo(self) -> bool:
        proceso = _PROCESOS.get(self.clave)
        if proceso is not None:
            return proceso.poll() is None
        pid = self._pid()
        return pid is not None and _pid_vivo(pid)

    def _borrar_salidas(self) -> None:
        if self.espacio.salidas.is_dir() and not self.espacio.salidas.is_symlink():
            shutil.rmtree(self.espacio.salidas)

    def _cerrar(self) -> None:
        """El proceso ya no existe: se quita el cerrojo. Sin un final correcto no queda ningún resultado."""
        if self.estado().get('estado') not in FINALES:
            registrar(self.espacio.ruta, 'el proceso de cálculo ha terminado sin anotar cómo')
            self._anotar(estado='error', motivo='el proceso terminó sin avisar')
        if self.estado().get('estado') != 'terminada':
            self._borrar_salidas()
        proceso = _PROCESOS.pop(self.clave, None)
        if proceso is not None and proceso.stdin is not None:
            proceso.stdin.close()
        self.cerrojo.unlink(missing_ok=True)

    def en_curso(self) -> bool:
        if not self.cerrojo.exists():
            return False
        if self._vivo():
            return True
        self._cerrar()
        return False

    def hay_resultados(self) -> bool:
        return (not self.en_curso() and self.estado().get('estado') == 'terminada'
                and self.espacio.carpeta_de_resultados() is not None)

    # ------------------------------------------------------------------ lanzar

    def huella_de(self, pasos, opciones: dict) -> str:
        """Identifica una petición: mismos archivos, mismas respuestas y mismas opciones."""
        familias = sorted(self.espacio.familias.iterdir()) if self.espacio.familias.is_dir() else []
        partes = {'ventas': {p.name: huella(p) for p in self.espacio.archivos_de_ventas()},
                  'familias': {p.name: huella(p) for p in familias},
                  'guion': [p.a_dict() for p in pasos], 'opciones': opciones}
        return hashlib.sha256(json.dumps(partes, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()

    def lanzar(self, pasos, opciones: dict | None = None) -> str:
        """Devuelve 'lanzada', 'ya_en_curso' o 'ya_calculada'."""
        pasos, opciones = list(pasos), dict(opciones or {})
        if self.en_curso():
            return 'ya_en_curso'
        peticion = self.huella_de(pasos, opciones)
        if self.hay_resultados() and self.estado().get('huella') == peticion:
            return 'ya_calculada'
        try:
            descriptor = os.open(self.cerrojo, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return 'ya_en_curso'
        try:
            self.espacio.comprobar_codigo()
            self._borrar_salidas()
            for nombre in REGISTROS:
                (self.espacio.ruta / nombre).unlink(missing_ok=True)
            for respuesta in self.espacio.ruta.glob('respuesta_*.json'):
                respuesta.unlink()
            (self.espacio.ruta / 'peticion.json').write_text(
                json.dumps({'guion': [p.a_dict() for p in pasos], 'opciones': opciones}, ensure_ascii=False, indent=1),
                encoding='utf-8')
            self._anotar(estado='en_curso', huella=peticion, inicio=time.time())
            entorno = os.environ | {'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1',
                                    'TFG_VIGILAR': '1', 'TFG_PADRE': str(os.getpid())}
            entorno.pop('PYTHONPATH', None)
            # En Linux, la entrada estándar es un conducto que nunca se usa para escribir: si esta aplicación se cierra,
            # el conducto se cierra y el proceso de cálculo se detiene solo. En Windows no se puede hacer así: una
            # lectura pendiente en un conducto bloquea cualquier otra operación sobre él, también en los procesos
            # que lo heredan, y el programa se quedaba parado al arrancar. Allí la entrada es nula y el proceso
            # vigila directamente a esta aplicación por su número de proceso (ver `lanzador.py`).
            if os.name == 'nt':
                aparte = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP, 'stdin': subprocess.DEVNULL}
            else:
                aparte = {'start_new_session': True, 'stdin': subprocess.PIPE}
            with open(self.espacio.ruta / 'pantalla.txt', 'w', encoding='utf-8') as pantalla, \
                    open(self.espacio.ruta / 'errores.txt', 'w', encoding='utf-8') as errores:
                proceso = subprocess.Popen([sys.executable, '-W', 'ignore', str(LANZADOR), str(self.espacio.ruta)],
                                           cwd=self.espacio.ruta, stdout=pantalla, stderr=errores, env=entorno, **aparte)
            registrar(self.espacio.ruta, f'lanza el proceso {proceso.pid} (modo {opciones.get("modo", "cálculo")}); '
                                         f'entrada estándar {"nula" if os.name == "nt" else "conducto"}')
            os.write(descriptor, str(proceso.pid).encode('ascii'))
            (self.espacio.ruta / 'ultimo_proceso.txt').write_text(str(proceso.pid), encoding='ascii')
            _PROCESOS[self.clave] = proceso
        except BaseException:
            os.close(descriptor)
            self.cerrojo.unlink(missing_ok=True)
            raise
        os.close(descriptor)
        return 'lanzada'

    def esperar(self, segundos: float | None = None) -> dict:
        proceso = _PROCESOS.get(self.clave)
        if proceso is not None:
            try:
                proceso.wait(timeout=segundos)
            except subprocess.TimeoutExpired:
                return self.estado()
        self.en_curso()
        return self.estado()

    # ------------------------------------------------------------------ cancelar

    def cancelar(self, espera: float = 5.0) -> bool:
        """Termina el proceso y los que haya abierto, y borra lo que hubiera escrito. False si no había nada en curso."""
        if not self.cerrojo.exists():
            return False
        pid, proceso = self._pid(), _PROCESOS.get(self.clave)
        registrar(self.espacio.ruta, f'cancela el proceso {pid}')
        if pid is not None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True, **NULA)
            else:
                for senal in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(pid, senal)
                    except (ProcessLookupError, PermissionError):
                        break
                    limite = time.time() + espera
                    while time.time() < limite:
                        if proceso is not None:
                            proceso.poll()
                        if not _grupo_vivo(pid):
                            break
                        time.sleep(0.1)
                    if not _grupo_vivo(pid):
                        break
        if proceso is not None:
            try:
                proceso.wait(timeout=espera)
            except subprocess.TimeoutExpired:
                pass
        self._anotar(estado='cancelada')
        self._cerrar()
        return True

    # ------------------------------------------------------------------ bloqueos

    def sin_actividad(self) -> dict:
        """Cuánto lleva el cálculo sin hacer nada que se note: sin gastar procesador y sin escribir en ninguno de sus
        registros. Para avisar de un bloqueo en lugar de esperar sin fin.
        {'segundos': float, 'con_procesador': si se ha podido medir el procesador, 'etapa': última etapa anotada}"""
        if not self.en_curso():
            _ACTIVIDAD.pop(self.clave, None)
            return {'segundos': 0.0, 'con_procesador': False, 'etapa': self.ultima_etapa()}
        ahora, pid = time.time(), self._pid()
        cpu = _cpu_del_arbol(pid) if pid is not None else None
        firma = []
        for nombre in VIGILADOS:
            try:
                datos = (self.espacio.ruta / nombre).stat()
                firma.append((nombre, datos.st_size, datos.st_mtime_ns))
            except OSError:
                firma.append((nombre, None, None))
        inicio = self.estado().get('inicio')
        ref = _ACTIVIDAD.get(self.clave)
        cambio = (ref is None or ref['inicio'] != inicio or ref['firma'] != firma or (ref['cpu'] is None) != (cpu is None)
                  or (cpu is not None and cpu - ref['cpu'] > 0.2))
        if cambio:
            if ref is None or ref['inicio'] != inicio:
                # Primera vez que se mira: se cuenta desde lo último que se escribió, o desde que empezó.
                escrito = [m / 1e9 for _, _, m in firma if m is not None]
                desde = max([float(inicio or ahora)] + escrito)
            else:
                desde = ahora
            ref = _ACTIVIDAD[self.clave] = {'inicio': inicio, 'firma': firma, 'cpu': cpu, 'desde': min(desde, ahora)}
        return {'segundos': ahora - ref['desde'], 'con_procesador': cpu is not None, 'etapa': self.ultima_etapa()}

    def seguir_esperando(self) -> None:
        """El usuario prefiere esperar: la cuenta vuelve a empezar."""
        _ACTIVIDAD.pop(self.clave, None)
        registrar(self.espacio.ruta, 'el usuario decide seguir esperando')

    def ultima_etapa(self) -> str:
        """La última etapa que anotó el proceso de cálculo, sin la hora."""
        for linea in reversed(self.diagnostico().splitlines()):
            if '[lanzador' in linea:
                return linea.split('] ', 1)[-1]
        return ''

    def diagnostico(self, lineas: int = 80) -> str:
        """Las últimas líneas de `diagnostico.txt` y de `pilas.txt`. Solo etapas y líneas de código, ningún dato."""
        partes = []
        for nombre in (DIAGNOSTICO, PILAS):
            try:
                texto = (self.espacio.ruta / nombre).read_text(encoding='utf-8', errors='replace').splitlines()
            except OSError:
                continue
            if any(l.strip() and not l.startswith('---') for l in texto):
                partes.append(f'== {nombre} ==\n' + '\n'.join(texto[-lineas:]))
        return '\n\n'.join(partes)

    def procesos_vivos(self) -> bool:
        """Para comprobar una cancelación: ¿queda algún proceso del cálculo?"""
        try:
            pid = int((self.espacio.ruta / 'ultimo_proceso.txt').read_text(encoding='ascii'))
        except (OSError, ValueError):
            return False
        return _grupo_vivo(pid) if os.name != 'nt' else _pid_vivo(pid)
