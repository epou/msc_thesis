from configparser import ConfigParser, ExtendedInterpolation
import os

CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../settings.ini')

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(CONFIG_FILE)

config.set("Common", "root_dir", os.path.realpath(os.path.dirname(CONFIG_FILE)))
