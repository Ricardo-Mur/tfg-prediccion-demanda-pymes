"""Proceso que ejecuta el programa congelado dentro de un espacio de ejecución, con las respuestas del guion.

Uso:  python interfaz/lanzador.py <espacio>

No modifica el programa. Hace tres cosas en este proceso, y solo en este proceso:
  - carga `predecir_v2` desde el espacio (nunca desde el proyecto) y lo ejecuta con su función `main`;
  - sustituye el teclado por el lector del guion, que comprueba cada pregunta antes de contestarla;
  - rechaza cualquier conexión que no sea local: el cálculo no necesita la red.

Deja en el espacio `estado.json` con cómo ha terminado. La pantalla del programa va a la salida estándar, que
quien lo lanza guarda en `pantalla.txt`.

Para saber dónde se detiene si algo falla, deja además dos registros en el espacio. Ninguno lleva datos de los
archivos de ventas ni texto de las preguntas:
  - `diagnostico.txt`: cada etapa (arranque, vigilancia, carga del programa, cada pregunta y respuesta, final), con la
    hora. Cada línea se escribe y se guarda en el disco en el momento, para poder leerla aunque el proceso se quede parado;
  - `pilas.txt`: si el proceso pasa un rato sin llegar a la siguiente etapa, en qué línea de código está cada hilo.
"""
import builtins
import faulthandler
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

AQUI = Path(__file__).resolve().parent
LOCALES = ('127.0.0.1', '::1', 'localhost')
ANTES_DE_CALCULAR = 'CALCULANDO LA VERSIÓN 2'      # lo que escribe el programa cuando ya no va a preguntar nada más
ESPERA_MAXIMA = 4 * 3600                           # segundos esperando una respuesta
DIAGNOSTICO, PILAS = 'diagnostico.txt', 'pilas.txt'
PILAS_PREGUNTAS = float(os.environ.get('TFG_PILAS_SEGUNDOS', 30))   # sin llegar a la siguiente pregunta
PILAS_CALCULO = 120.0                                               # durante el cálculo
_INICIO = time.time()
_PILAS = None


def registrar(texto: str) -> None:
    """Una línea en `diagnostico.txt`: escrita, cerrada y guardada en el disco en el acto. Nunca falla."""
    espacio = os.environ.get('TFG_ESPACIO')
    if not espacio:
        return
    try:
        with open(Path(espacio) / DIAGNOSTICO, 'a', encoding='utf-8') as f:
            f.write(f'{time.strftime("%H:%M:%S")} +{time.time() - _INICIO:7.1f} s  [lanzador {os.getpid()}] {texto}\n')
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass


def anotar_pilas_cada(segundos: float | None) -> None:
    """Si pasan `segundos` sin volver a llamar a esta función, se anota en `pilas.txt` en qué línea de código está cada
    hilo, y se repite cada `segundos`. Solo nombres de archivos, funciones y líneas. None: se deja de anotar."""
    faulthandler.cancel_dump_traceback_later()
    if segundos and _PILAS is not None:
        faulthandler.dump_traceback_later(segundos, repeat=True, file=_PILAS)


def tipo_de_entrada() -> str:
    """Qué es la entrada estándar de este proceso. Se mira antes de empezar a vigilar."""
    import stat
    try:
        modo = os.fstat(0).st_mode
    except OSError:
        return 'ninguna'
    if stat.S_ISFIFO(modo):
        return 'conducto'
    if stat.S_ISCHR(modo):
        return 'dispositivo (nula o consola)'
    return 'archivo' if stat.S_ISREG(modo) else 'otra'


def bloquear_red() -> None:
    """Cualquier conexión de red que no sea con este mismo ordenador se rechaza y se anota."""
    if getattr(socket.socket, '_tfg_sin_red', False):
        return
    conectar, conectar_ex = socket.socket.connect, socket.socket.connect_ex

    def comprobar(self, direccion):
        if self.family in (socket.AF_INET, socket.AF_INET6) and str(direccion[0]) not in LOCALES:
            espacio = os.environ.get('TFG_ESPACIO')
            if espacio:
                with open(Path(espacio) / 'intentos_de_red.txt', 'a', encoding='utf-8') as f:
                    f.write('intento de conexión rechazado\n')
            raise ConnectionRefusedError('La aplicación no abre conexiones con el exterior.')

    def connect(self, direccion):
        comprobar(self, direccion)
        return conectar(self, direccion)

    def connect_ex(self, direccion):
        comprobar(self, direccion)
        return conectar_ex(self, direccion)

    socket.socket.connect, socket.socket.connect_ex, socket.socket._tfg_sin_red = connect, connect_ex, True


bloquear_red()          # también en los procesos de cálculo, que vuelven a importar este archivo


class Pantalla:
    """Lo que escribe el programa sigue su camino y, además, se recuerda lo escrito desde la última pregunta."""

    def __init__(self, destino, parar_en: str = '', parar=None):
        self.destino, self.reciente, self.parar_en, self.parar = destino, [], parar_en, parar

    def write(self, texto):
        if self.parar_en and self.parar_en in texto:
            raise self.parar            # modo de preguntas: el programa ya no pregunta más y va a calcular
        self.reciente.append(texto)
        return self.destino.write(texto)

    def flush(self):
        self.destino.flush()

    def desde_la_ultima_pregunta(self) -> str:
        texto = ''.join(self.reciente)
        self.reciente.clear()
        return texto

    def __getattr__(self, nombre):
        return getattr(self.destino, nombre)


def vigilar_a_quien_lanzo() -> None:
    """Si la aplicación que lanzó este proceso desaparece, el cálculo se detiene, con sus procesos.

    En Linux, la aplicación deja abierto un conducto hacia la entrada estándar y no escribe nunca en él: cuando se
    cierra, la lectura termina. En Windows no se puede leer así (una lectura pendiente en un conducto bloquea
    cualquier otra operación sobre él, y el programa se quedaba parado al arrancar): se espera a que termine el
    proceso de la aplicación, por su número, que llega en TFG_PADRE."""
    def detener():
        registrar('vigilancia: la aplicación que lanzó el cálculo ya no existe; se detiene el cálculo')
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(os.getpid()), '/T', '/F'], capture_output=True, stdin=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgrp(), signal.SIGTERM)      # este proceso y los de cálculo: forman un grupo propio
        os._exit(6)

    def esperar_conducto():
        try:
            sys.stdin.buffer.read()
        except Exception:
            pass
        detener()

    def esperar_a_la_aplicacion(pid: int):
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel32.WaitForSingleObject.restype = wintypes.DWORD
            manejador = kernel32.OpenProcess(0x00100000, False, pid)        # SYNCHRONIZE: solo para esperarlo
            if not manejador:
                registrar(f'vigilancia: no se puede abrir el proceso {pid} de la aplicación (error {ctypes.get_last_error()}); '
                          'sigue sin vigilancia')
                return              # la limpieza de sesiones abandonadas se encarga después
            registrar(f'vigilancia: espera a que termine el proceso {pid} de la aplicación')
            if kernel32.WaitForSingleObject(manejador, 0xFFFFFFFF) == 0:       # 0 = el proceso ha terminado
                detener()
        except Exception as e:
            registrar(f'vigilancia: error {type(e).__name__}; sigue sin vigilancia')

    if os.name == 'nt':
        padre = os.environ.get('TFG_PADRE', '')
        if padre.isdigit():
            threading.Thread(target=esperar_a_la_aplicacion, args=(int(padre),), daemon=True).start()
        else:
            registrar('vigilancia: no llega el número de proceso de la aplicación (TFG_PADRE); sin vigilancia')
    else:
        registrar('vigilancia: espera a que se cierre el conducto de la entrada estándar')
        threading.Thread(target=esperar_conducto, daemon=True).start()


def escribir_json(ruta: Path, datos) -> None:
    temporal = ruta.with_suffix('.tmp')
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding='utf-8')
    for _ in range(99):             # en Windows falla si la interfaz tiene abierto el archivo justo en ese momento
        try:
            os.replace(temporal, ruta)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(temporal, ruta)


def anotar(espacio: Path, **datos) -> None:
    ruta = espacio / 'estado.json'
    actual = json.loads(ruta.read_text(encoding='utf-8')) if ruta.exists() else {}
    escribir_json(ruta, actual | datos)
    if 'estado' in datos:
        registrar(f'final: {datos["estado"]}' + (f' ({datos["motivo"]})' if datos.get('motivo') else ''))


def preguntar(espacio: Path, n: int, mensaje: str, escrito: str) -> str:
    """Deja la pregunta n.º `n` para la interfaz y espera su respuesta, que debe ser para esa misma pregunta."""
    from guion import DetenerEjecucion
    anotar_pilas_cada(None)                             # mientras espera al usuario, no está parado
    escribir_json(espacio / 'pregunta.json', {'n': n, 'pregunta': str(mensaje), 'contexto': escrito})
    registrar(f'pregunta {n}: en la interfaz, esperando la respuesta')
    ruta, limite = espacio / f'respuesta_{n}.json', time.time() + ESPERA_MAXIMA
    while not ruta.exists():
        if time.time() > limite:
            raise DetenerEjecucion('sin_respuesta', 'ha pasado demasiado tiempo sin respuesta')
        time.sleep(0.05)
    dato = None
    for _ in range(100):            # en Windows, la interfaz puede estar terminando de sustituir el archivo
        try:
            dato = json.loads(ruta.read_text(encoding='utf-8'))
            break
        except (PermissionError, ValueError):
            time.sleep(0.05)
    if dato is None:
        dato = json.loads(ruta.read_text(encoding='utf-8'))
    registrar(f'pregunta {n}: respuesta recibida')
    anotar_pilas_cada(PILAS_PREGUNTAS)
    if dato.get('n') != n or not isinstance(dato.get('respuesta'), str):
        raise DetenerEjecucion('pregunta_inesperada', 'la respuesta recibida no es de esta pregunta')
    return dato['respuesta']


def cargar_programa(espacio: Path):
    """`predecir_v2`, cargado desde el espacio. Si se cargara desde otro sitio, no se ejecuta."""
    programa = (espacio / 'programa').resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(programa))
    os.chdir(programa)
    registrar('carga del programa: importando predecir_v2 y sus librerías')
    antes = time.time()
    import predecir_v2
    import src
    registrar(f'carga del programa: importado en {time.time() - antes:.1f} s')
    for modulo in (predecir_v2, src):
        if programa not in Path(modulo.__file__).resolve().parents:
            raise RuntimeError('El programa no se ha cargado desde el espacio de ejecución.')
    return predecir_v2


def main() -> int:
    global _PILAS
    espacio = Path(sys.argv[1]).resolve()
    os.environ['TFG_ESPACIO'] = str(espacio)
    registrar(f'arranque: Python {sys.version.split()[0]} en {sys.platform}; ejecutable {sys.executable}; '
              f'proceso padre {os.getppid()}; aplicación {os.environ.get("TFG_PADRE", "-")}; entrada estándar: {tipo_de_entrada()}')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)     # `pantalla.txt` al día, línea a línea
    _PILAS = open(espacio / PILAS, 'a', encoding='utf-8')
    _PILAS.write(f'--- lanzador {os.getpid()}, {time.strftime("%H:%M:%S")} ---\n')
    _PILAS.flush()
    if os.environ.get('TFG_VIGILAR') == '1':
        vigilar_a_quien_lanzo()
    else:
        registrar('vigilancia: desactivada (sin TFG_VIGILAR)')
    sys.path.insert(0, str(AQUI))
    from guion import DetenerEjecucion, Lector, LectorDePreguntas, Paso

    peticion = json.loads((espacio / 'peticion.json').read_text(encoding='utf-8'))
    opciones = peticion.get('opciones', {})
    solo_preguntas = opciones.get('modo') == 'preguntas'
    registrar(f'petición: modo {"preguntas" if solo_preguntas else "cálculo"}; {len(peticion["guion"])} respuestas ya dadas; '
              f'{opciones.get("procesos", 1)} procesos')
    anotar_pilas_cada(PILAS_PREGUNTAS if solo_preguntas else PILAS_CALCULO)
    pantalla = Pantalla(sys.stdout, ANTES_DE_CALCULAR if solo_preguntas else '', DetenerEjecucion('preguntas_completas'))
    sys.stdout = pantalla
    pasos = [Paso.desde_dict(d) for d in peticion['guion']]
    if solo_preguntas:
        lector = LectorDePreguntas(pasos, pantalla.desde_la_ultima_pregunta, lambda n, m, e: preguntar(espacio, n, m, e),
                                   lambda grabados: escribir_json(espacio / 'grabado.json', [p.a_dict() for p in grabados]))
    else:
        lector = Lector(pasos, pantalla.desde_la_ultima_pregunta)

    def teclado(mensaje=''):
        respuesta = lector(mensaje)
        pantalla.destino.write(f'{mensaje}{respuesta}\n')      # la pregunta y su respuesta quedan en el registro
        return respuesta

    builtins.input = teclado
    try:
        if os.environ.get('TFG_PRUEBA_BLOQUEO') == '1':     # solo lo usa `prueba_windows.py`: simula un bloqueo
            registrar('PRUEBA: bloqueo simulado antes de cargar el programa')
            threading.Event().wait()
        predecir_v2 = cargar_programa(espacio)
        if solo_preguntas and opciones.get('revision'):
            # Revisión agrupada: copias de solo lectura de lo que reciben y devuelven las preguntas (`observadores.py`).
            # Solo aquí, en el modo de preguntas: el cálculo se lanza aparte, sin ellos.
            import observadores
            observadores.instalar(predecir_v2, espacio / 'revision', registrar)
        familias = espacio / 'familias' / opciones['familias'] if opciones.get('familias') else None
        registrar('programa: empieza main()')
        informe = predecir_v2.main(int(opciones.get('procesos', 1)), None, familias,
                                   opciones.get('columna_producto'), opciones.get('columna_familia'))
        lector.comprobar_final()
    except DetenerEjecucion as e:
        if solo_preguntas and e.motivo == 'preguntas_completas':
            anotar(espacio, estado='preguntas_completas', preguntas=lector.usadas)
            return 0
        anotar(espacio, estado='detenida', motivo=e.motivo, detalle=e.detalle, preguntas=lector.usadas)
        return 3
    except BaseException as e:      # también una interrupción: queda anotado cómo terminó
        registrar(f'error: {type(e).__name__}')
        (espacio / 'error_tecnico.txt').write_text(traceback.format_exc(), encoding='utf-8')
        anotar(espacio, estado='error', motivo=type(e).__name__, preguntas=lector.usadas)
        return 4
    finally:
        anotar_pilas_cada(None)
        sys.stdout.flush()
    if informe is None:
        anotar(espacio, estado='sin_resultados', preguntas=lector.usadas)
        return 5
    anotar(espacio, estado='terminada', preguntas=lector.usadas)
    return 0


if __name__ == '__main__':
    sys.exit(main())
