"""FruitCraft: StarCraft Brood War (OpenSnowstorm) driven by FruitLoop fly brains.

Layers:
  _engine        pybind11 bridge over mini-openbwapi (one game per process)
  engine         BroodWarGame: match lifecycle, observations, commands
  micro          spawned micro-combat scenarios with a scripted opponent
  fly.*          encoder / brain pool / decoder / controller (one fly per unit)
"""

__version__ = "0.1.0"
