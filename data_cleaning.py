import pandas as pd

DATA_PATH = "data/Music Info.csv"

def clean_data(data):
    data = data.drop_duplicates(subset='spotify_id').drop(columns = ['genre', 'spotify_id']).fillna({'tags':'no_tags'})
    data['name'] = data['name'].str.lower().str.strip()
    data['artist'] = data['artist'].str.lower().str.strip()
    data['tags'] = data['tags'].str.lower().str.strip()
    return data.reset_index(drop = True)

def data_for_content_filtering(data):
    return data.drop(columns = ["track_id","name","spotify_preview_url"])

def main(data_path):
    data = pd.read_csv(data_path)
    cleaned_data = clean_data(data)
    cleaned_data.to_csv('data/cleaned_data.csv', index=False)
    
if __name__ == '__main__':
    main(DATA_PATH)
    
