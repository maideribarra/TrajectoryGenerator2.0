from dataclasses import dataclass

@dataclass
class MisDatos:
    NUM_SEQ: int
    INPUT_DIM: int
    OUTPUT_DIM: int
    HID_DIM: int

datos = MisDatos(3,4,6,7)
import yaml
from typing import Any
class Datos:
    def __init__(self) -> None:
        pass
    def setter(self, nombre: str, valor: Any):
        exec(f"self.{nombre}={valor}")

with open("/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/model/Autoencoder LSTM v2/experimentos/exp4/exp4.yaml", "rb") as f:
    datos = yaml.load(f, yaml.Loader)

mi_clase = Datos()
for nombre, dato in datos.items():
    mi_clase.setter(nombre,dato)

print()