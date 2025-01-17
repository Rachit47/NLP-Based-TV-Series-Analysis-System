import pandas as pd
from glob import glob

def load_subtitles_dataset(dataset_path):
    subtitles_paths = glob(dataset_path+'/*.ass')
    
    scripts = []
    episode_num = []
    for path in subtitles_paths:
        # Read lines
        with open(path, 'r', encoding = 'utf-8', errors = 'ignore') as file:
            lines = file.readlines()
            lines = lines[27:] # before line 27, its all meta data (which is not required)
            lines = [",".join(line.split(',')[9:]) for line in lines] # we have removed everything before 9th comma and include only the text after 9th comma
            
        lines = [line.replace('\\N', ' ') for line in lines]
        script = " ".join(lines)
        
        episode = int(path.split('-')[-1].split('.')[0].strip())
        
        scripts.append(script)
        episode_num.append(episode)
    df = pd.DataFrame.from_dict({'episode':episode_num, "script": scripts})
    return df