from pathlib import Path
import requests
root=Path('models/grounding-dino-local');root.mkdir(parents=True,exist_ok=True)
for name in ['config.json','model.safetensors']:
    target=root/name
    with requests.get('https://huggingface.co/IDEA-Research/grounding-dino-tiny/resolve/main/'+name,stream=True,timeout=(30,60)) as response:
        response.raise_for_status()
        with target.open('wb') as f:
            total=0
            for chunk in response.iter_content(1024*1024):
                f.write(chunk);total+=len(chunk)
                if total%(50*1024*1024)<1024*1024:print(name,total,flush=True)
    print('Downloaded',name,target.stat().st_size,flush=True)
