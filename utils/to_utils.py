class DictToObject:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            elif isinstance(value, list):
                setattr(self, key, [
                    DictToObject(item) if isinstance(item, dict) else item
                    for item in value
                ])
            else:
                setattr(self, key, value)