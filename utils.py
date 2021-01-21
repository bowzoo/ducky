import json
import yaml


def yml_to_json_str(file_path):
    with open(file_path, 'r') as yaml_in:
        yaml_object = yaml.safe_load(yaml_in) # yaml_object will be a list or a dict
        json_str = json.dumps(yaml_object)

    return json_str
