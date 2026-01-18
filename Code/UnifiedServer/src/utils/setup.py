import json
import os

def get_classes(): 
    try: 
        # Calculate path relative to this file (UnifiedServer/src/utils/setup.py -> UnifiedServer/src/config.json)
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_path, 'config.json')
        
        with open(config_path) as f: 
            config = json.load(f) 
        classes = config['classes']
        return classes         
    except Exception as e: 
        print(f'Something went wrong loading your config file: {e}')
        return []

def get_colors(): 
    try: 
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_path, 'config.json')
        
        with open(config_path) as f: 
            config = json.load(f) 
        classes = config['classes'] 
        colors = config['colors']
        return colors
    except Exception as e: 
        print(f'Something went wrong loading your config file: {e}')
        return []
