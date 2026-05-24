import os, json, sys

config_name = os.environ.get('CONFIG_NAME', 'config_ipazia')
conf_path = f'conf/{config_name}.yaml'

mode = 'RUN'
graph_types = []
encoders = []

try:
    with open(conf_path) as f:
        lines = f.readlines()
        
    # Detect mode
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'): continue
        if stripped.startswith('mode:'):
            mode = stripped.split(':', 1)[1].split('#')[0].strip().upper()
            break
            
    if mode == 'MULTIRUN':
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'): continue
            if 'data.graph_type:' in stripped:
                val = stripped.split(':', 1)[1].split('#')[0].strip()
                graph_types = [x.strip() for x in val.split(',')]
            elif 'model.encoder:' in stripped:
                val = stripped.split(':', 1)[1].split('#')[0].strip()
                encoders = [x.strip() for x in val.split(',')]
    else:
        # RUN mode (single run)
        in_data = False
        in_model = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'): continue
            
            if stripped.startswith('data:'):
                in_data = True
                in_model = False
            elif stripped.startswith('model:'):
                in_model = True
                in_data = False
            elif stripped.startswith('evaluation:') or stripped.startswith('output:'):
                in_data = False
                in_model = False
                
            if in_data and stripped.startswith('graph_type:'):
                val = stripped.split(':', 1)[1].split('#')[0].strip()
                graph_types = [val]
            if in_model and stripped.startswith('encoder:'):
                val = stripped.split(':', 1)[1].split('#')[0].strip()
                encoders = [val]
except Exception as e:
    print(f'Error reading config: {e}')
    sys.exit(1)
    
# Fallbacks if parsing fails
if not graph_types: graph_types = ['reply']
if not encoders: encoders = ['gcn']

print(f'Mode: {mode}')
print(f'Found Encoders: {encoders}')
print(f'Found Graph Types: {graph_types}')

with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
    f.write(f'graph_types={json.dumps(graph_types)}\n')
    f.write(f'encoders={json.dumps(encoders)}\n')
