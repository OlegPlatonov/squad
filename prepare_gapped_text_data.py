import numpy as np
import pandas as pd
import spacy
import re
import os
import random
from multiprocessing import Pool
from functools import partial


nlp = spacy.blank('en')


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc if not token.text.isspace()]


def get_raw_texts(file):
    texts = []

    with open(file, encoding='utf8') as file:
        for line in file.readlines():
            line = line.replace("'", '"').lower().strip()
            if line.startswith('<doc'):  # begin new document
                current_text = []
                skip_next = True
            elif skip_next:  # skip title
                skip_next = False
            elif line == '</doc>':  # end document
                texts.append(current_text)
            else:
                if line != '':
                    line = process_raw_line(line)
                    if line != '':
                        current_text.append(line)

    return texts


def process_raw_line(line):
    line = re.sub('<ref.*/ref>', '', line, flags=re.DOTALL)
    line = re.sub('<ref.*">', '', line, flags=re.DOTALL)
    line = re.sub('<blockquote>', '', line)
    line = re.sub('</blockquote>', '', line)
    line = re.sub('<sub.*/sub>', '', line, flags=re.DOTALL)
    line = re.sub('<sup.*/sup>', '', line, flags=re.DOTALL)
    line = re.sub('<li.*/li>', '', line, flags=re.DOTALL)
    line = re.sub('<.*>', '', line, flags=re.DOTALL)
    line = re.sub('\(.*?\)', '', line, flags=re.DOTALL)
    line = re.sub('; ,', ',', line)
    line = re.sub(', ;', ',', line)
    line = re.sub('[\s\s+]', ' ', line)
    return line


def simple_split(string):
    raw_split = re.split('([.?!,:;])', string)
    raw_split = [phrase for phrase in raw_split if phrase != '']
    split = [raw_split[0]]
    for i in range(1, len(raw_split)):
        if len(split) > 0 and split[-1][-1].isdigit() and i+1 < len(raw_split) and raw_split[i+1][0].isdigit():
            digit = split.pop(-1)
            raw_split[i+1] = digit + raw_split[i] + raw_split[i+1]
        elif raw_split[i] in '.?!,:;':
            split[-1] += raw_split[i]
        else:
            split.append(raw_split[i])

    split = [phrase.strip() for phrase in split]
    return split


def make_sentences(splitted_text):
    def split_sentence(sentence):
        new_sentences = []
        step = 12 if len(sentence) >= 22 else 10
        indices = list(range(0, len(sentence), step))
        for i, idx in enumerate(indices):
            new_sentence = sentence[idx:idx + step]
            if new_sentence[-1] == '-' and i + 1 < len(indices):
                new_sentence = new_sentence[:-2]
                indices[i + 1] -= 2
            if len(new_sentence) < 8 and len(new_sentences) > 0:
                new_sentences[-1] += new_sentence
            else:
                new_sentences.append(new_sentence)
        return new_sentences

    sentences = []
    current_sentence = []
    for i in range(len(splitted_text)):
        if len(current_sentence) < 10 or len(splitted_text[i]) < 5:
            current_sentence += splitted_text[i]
        else:
            if len(current_sentence) < 20:
                sentences.append(current_sentence)
            else:
                new_sentences = split_sentence(current_sentence)
                for new_sentence in new_sentences:
                    sentences.append(new_sentence)
            current_sentence = splitted_text[i]

    if len(current_sentence) < 8 and len(sentences) > 0:
        sentences[-1] += current_sentence
    else:
        sentences.append(current_sentence)

    return sentences


def process_all(num_processes,
                load_path='D:\Oleg\Wikipedia_corpus\Data\Extracted',
                save_path='D:\Oleg\Wikipedia_corpus\Data\Gapped_Text\Text',
                text_size=25,
                num_gaps=4,
                min_space=2,
                num_random_sent=2):
    folders = [os.path.join(load_path, folder) for folder in os.listdir(load_path)]

    folder_processer = partial(process_folder,
                               save_path=save_path,
                               text_size=text_size,
                               num_gaps=num_gaps,
                               min_space=min_space,
                               num_random_sent=num_random_sent)

    with Pool(num_processes) as pool:
        pool.map(folder_processer, folders)


def process_folder(folder, save_path, text_size=25, num_gaps=4, min_space=2, num_random_sent=2):
    files = [os.path.join(folder, file) for file in os.listdir(folder)]
    folder = os.path.basename(os.path.normpath(folder))

    data = {
        'text': [],
        'fragments': [],
        'gap_indices': [],
        'correct_gaps': []
    }

    for file in files:
        process_file(file, data, chunk_size=text_size, num_gaps=num_gaps, min_space=min_space, num_random_sent=num_random_sent)

    data = pd.DataFrame(data)
    data.to_csv(os.path.join(save_path, f'{folder}.csv'), index=False)
    print(f'Finished processing data from {folder}.')


def process_file(file, data, chunk_size=25, num_gaps=4, min_space=2, num_random_sent=2):
    texts = get_raw_texts(file)
    for text in texts:
        text_splitted = []
        for para in text:
            text_splitted += simple_split(para)
        text_splitted = [word_tokenize(phrase) for phrase in text_splitted]
        sentences = make_sentences(text_splitted)
        if len(sentences) > chunk_size + 3 * num_random_sent:
            for gapped_text in process_text(sentences,
                                            chunk_size=chunk_size,
                                            num_gaps=num_gaps,
                                            min_space=min_space,
                                            num_random_sent=num_random_sent):
                text, fragments, gap_indices, correct_gaps = gapped_text
                data['text'].append(text)
                data['fragments'].append('|-|-|'.join(fragments))
                data['gap_indices'].append(' '.join(map(str, gap_indices)))
                data['correct_gaps'].append(' '.join(map(str, correct_gaps)))


def process_text(text, chunk_size=25, num_gaps=4, min_space=2, num_random_sent=2):
    chunk_bounds = list(range(0, len(text), chunk_size))
    for i in range(1, len(chunk_bounds)):
        chunk = text[chunk_bounds[i - 1]:chunk_bounds[i]]
        gap_sent_ids = get_ramdom_gap_ids(chunk_size=chunk_size, num_gaps=num_gaps, min_space=min_space)

        chunk = [['--START--']] + chunk
        gap_sent_ids += 1

        sent_lengths = [len(sent) for sent in chunk]

        fragments = []
        gap_token_ids = [0]
        for idx in gap_sent_ids:
            fragments.append(' '.join(['--START--'] + chunk[idx]))
            chunk[idx] = ['--GAP--']
            sent_lengths[idx] = 1
            gap_token_ids.append(sum(sent_lengths[:idx]))
        correct_gaps = list(range(1, num_gaps + 1))

        nonchunk_sent_ids = get_random_nonchunk_sent_ids(text_len=len(text),
                                                         chunk_size=chunk_size,
                                                         chunk_start=chunk_bounds[i - 1],
                                                         num_random_sent=num_random_sent)

        for idx in nonchunk_sent_ids:
            fragments.append(' '.join(['--START--'] + text[idx]))
            correct_gaps.append(0)

        chunk = ' '.join((' '.join(phrase) for phrase in chunk))
        yield chunk, fragments, gap_token_ids, correct_gaps


def get_ramdom_gap_ids(chunk_size, num_gaps, min_space):
    ids = np.array(sorted(random.sample(range(chunk_size - min_space * (num_gaps+1)), k=num_gaps)))
    ids += np.arange(1, num_gaps+1) * min_space
    return ids


def get_random_nonchunk_sent_ids(text_len, chunk_size, chunk_start, num_random_sent):
    ids = np.array(sorted(random.sample(range(text_len - chunk_size), k=num_random_sent)))
    ids += (ids > chunk_start) * chunk_size
    return ids


if __name__ == '__main__':
    random.seed(111)
    np.random.seed(111)
    process_all(num_processes=6)
