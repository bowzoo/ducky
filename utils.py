import yaml


def yml_to_dict(file_path):
    with open(file_path, 'r') as yaml_in:
        yaml_object = yaml.safe_load(yaml_in) # yaml_object will be a list or a dict

    return yaml_object
