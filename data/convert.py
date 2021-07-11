import json
import argparse
import os
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scienceparse', type = str, required = True)
    parser.add_argument('--deepfigure', type = str, required = True)
    parser.add_argument('--output_dir', type = str, default = 'train/paper')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.deepfigure) as j:
        figures = json.load(j)['figures']
    with open(args.scienceparse) as j:
        paper = json.load(j)

    sections = paper['metadata']['sections']

    for figure in figures:
        if not figure['figure_type'] == 'Figure':
            continue
        fig_name = 'Figure {} '.format(figure['name'])
        mention = None
        text = str()
        for section in sections:
            text += section['text']+'\n'
        paragraphs = text.split('\n')
        for paragraph in paragraphs:
            paragraph = paragraph.replace('Fig.', 'Figure')
            paragraph = paragraph.replace('Figures', 'Figure')
            paragraph = paragraph.replace('.', ' .')

            m = re.search(fig_name, paragraph)
            if not m is None:
                mention = paragraph.replace(' .', '.')

            if not mention is None:
                break         
        if not mention is None:
            break

        if not mention is None:
            data = {"caption":figure["caption_text"] , "mention":mention}
            with open(os.path.join(args.output_dir, 'Figure_{}.json'.format(figure['name'])), 'w') as j:
                json.dump(data, j)