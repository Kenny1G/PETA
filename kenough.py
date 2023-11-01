import torch
import argparse
import os
import matplotlib.pyplot as plt
import torchvision.utils
from PIL import Image
import numpy as np
from src.models import create_model
from src.utils.utils import create_dataloader, validate
import sqlite3
from datetime import date
from dateutil.rrule import rrule, DAILY


# ----------------------------------------------------------------------
# Parameters
parser = argparse.ArgumentParser(description='PETA: Photo album Event recognition using Transformers Attention.')
parser.add_argument('--model_path', type=str, default='./models_local/peta_32.pth')
parser.add_argument('--db_path', type=str, default='./photos.db')
parser.add_argument('--val_dir', type=str, default='./albums') #  /Graduation') # /0_92024390@N00')
parser.add_argument('--num_classes', type=int, default=23)
parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
parser.add_argument('--transformers_pos', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--transform_type', type=str, default='squish')
parser.add_argument('--album_sample', type=str, default='rand_permute')
parser.add_argument('--dataset_path', type=str, default='./data/ML_CUFED')
parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
parser.add_argument('--path_output', type=str, default='./outputs')
parser.add_argument('--use_transformer', type=int, default=1)
parser.add_argument('--album_clip_length', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.85)
parser.add_argument('--remove_model_jit', type=int, default=None)


# Function to get album from database
def get_album(args, date):
    # Connect to the SQLite database
    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()
    # Execute SQL query to fetch images taken on a specific date
    cursor.execute("SELECT FilePath FROM photos WHERE DateTaken LIKE ?", (date + "%",))
    # Get file paths from the query result
    files = [row[0] for row in cursor.fetchall()]
    conn.close()

    # If no images for a given date, return None
    if not files:
        return None, None

    # Update file paths to match the new directory structure
    files = ['/workspace' + file.replace('/Users/kennyosele/Documents/Projects/cs191/flask_app/static', '') for file in files]

    # The rest of the function remains the same
    n_files = len(files)
    idx_fetch = np.linspace(0, n_files-1, args.album_clip_length, dtype=int)
    tensor_batch = torch.zeros(len(idx_fetch), args.input_size, args.input_size, 3)
    for i, id in enumerate(idx_fetch):
        im = Image.open(files[id])
        im_resize = im.resize((args.input_size, args.input_size))
        np_img = np.array(im_resize, dtype=np.uint8)
        tensor_batch[i] = torch.from_numpy(np_img).float() / 255.0
    tensor_batch = tensor_batch.permute(0, 3, 1, 2).cuda()   # HWC to CHW
    montage = torchvision.utils.make_grid(tensor_batch).permute(1, 2, 0).cpu()
    return tensor_batch, montage


# Function to perform inference
def inference(tensor_batch, model, classes_list, args):

    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
    np_output = output.cpu().detach().numpy()
    idx_sort = np.argsort(-np_output)
    # Top-k
    detected_classes = np.array(classes_list)[idx_sort][: args.top_k]
    scores = np_output[idx_sort][: args.top_k]
    # Threshold
    idx_th = scores > args.threshold
    return detected_classes[idx_th], scores[idx_th]


# Function to display image
def display_image(im, tags, filename, path_dest):

    if not os.path.exists(path_dest):
        os.makedirs(path_dest)

    plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.axis('tight')
    plt.rcParams["axes.titlesize"] = 16
    plt.title("Predicted classes: {}".format(tags))
    plt.savefig(os.path.join(path_dest, filename))


# Function to update database with inference results
def update_database(args, date_str, tags):
    # Connect to the SQLite database
    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    # Insert tags into the dates_have_tags table
    for tag in tags:
        cursor.execute("INSERT INTO dates_have_tags (date, tag_id) SELECT ?, id FROM tags WHERE tag = ?", (date_str, tag))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


# Main function
def main():
    print('PETA demo of inference code on a single album.')

    # ----------------------------------------------------------------------
    # Preliminaries
    args = parser.parse_args()

    # Setup model
    print('creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    model = create_model(args).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    classes_list = np.array(list(state['idx_to_class'].values()))
    print('Class list:', classes_list)

    # Setup data loader
    print('creating data loader...')
    val_loader = create_dataloader(args)
    print('done\n')

    # Connect to the SQLite database
    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    # Create the tags and dates_have_tags tables
    cursor.execute("CREATE TABLE IF NOT EXISTS tags (id INTEGER PRIMARY KEY, tag TEXT UNIQUE)")
    cursor.execute("CREATE TABLE IF NOT EXISTS dates_have_tags (date TEXT, tag_id INTEGER, FOREIGN KEY(tag_id) REFERENCES tags(id))")

    # Populate the tags table
    for class_ in classes_list:
        cursor.execute("INSERT OR IGNORE INTO tags (tag) VALUES (?)", (class_,))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    # Loop over each date
    # for dt in rrule(DAILY, dtstart=date(2023, 1, 1), until=date.today()):
    for dt in rrule(DAILY, dtstart=date(2023, 1, 20), until=date(2023, 1, 21)):
        date_str = dt.strftime("%Y:%m:%d")
        print(date_str)
        tensor_batch, montage = get_album(args, date_str)

        # If no images for a given date, continue to next date
        if tensor_batch is None:
            continue

        # Inference
        tags, confs = inference(tensor_batch, model, classes_list, args)

        # Update database with inference results
        update_database(args, date_str, tags)

        # Visualization
        print(f"saving image to {os.path.join(args.path_output, 'date_results')} ")
        display_image(montage, tags, date_str + '.jpg', os.path.join(args.path_output, 'date_results'))

    print('Done\n')


if __name__ == '__main__':
    main()
