from kidneyClassifier.constants import *
from kidneyClassifier.utils.common import read_yaml_file

config_contents = read_yaml_file(CONFIG_FILE_PATH)

print(config_contents)