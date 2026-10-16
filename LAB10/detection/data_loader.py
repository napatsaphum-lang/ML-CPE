from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
ZIP_FILE = ROOT / 'archive (8).zip'
DATASET = ROOT / 'dataset'
CLASSES = ['ESP8266', 'Heltec_ESP32_Lora', 'Arduino_Nano', 'Raspberry_Pi_3']

def xml_to_yolo(xml_bytes):
    annotation = ET.fromstring(xml_bytes)
    size = annotation.find('size')
    width, height = float(size.findtext('width')), float(size.findtext('height'))
    labels = []
    for obj in annotation.findall('object'):
        name = obj.findtext('name')
        if name not in CLASSES:
            raise ValueError(f'Unknown class: {name}')
        box = obj.find('bndbox')
        xmin, ymin, xmax, ymax = [float(box.findtext(t)) for t in ('xmin','ymin','xmax','ymax')]
        xmin, xmax = max(0,min(xmin,width)), max(0,min(xmax,width))
        ymin, ymax = max(0,min(ymin,height)), max(0,min(ymax,height))
        if xmax <= xmin or ymax <= ymin:
            continue
        cx = ((xmin + xmax) / 2) / width
        cy = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        labels.append(f'{CLASSES.index(name)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    return '\n'.join(labels) + ('\n' if labels else '')

def prepare_dataset():
    yaml_path = DATASET / 'data.yaml'
    if yaml_path.exists():
        print('Using existing YOLO dataset:', DATASET)
        return yaml_path
    if not ZIP_FILE.exists():
        raise FileNotFoundError(f'Please copy archive (8).zip into: {ROOT}')
    with zipfile.ZipFile(ZIP_FILE) as z:
        items = z.namelist()
        def pairs(group):
            images = [n for n in items if f'/{group}/' in n and n.lower().endswith(('.jpg','.jpeg','.png'))]
            return [(n, n.rsplit('.',1)[0]+'.xml') for n in images]
        train_pairs, test_pairs = pairs('train'), pairs('test')
        if not train_pairs or not test_pairs:
            raise ValueError('Dataset must contain train/ and test/ JPEG images and XML annotations')
        random.Random(42).shuffle(train_pairs)
        val_count = max(1,round(len(train_pairs)*0.2))
        splits = {'train': train_pairs[val_count:], 'val':train_pairs[:val_count], 'test':test_pairs}
        for split, group in splits.items():
            image_dir = DATASET / split / 'images'
            label_dir = DATASET / split / 'labels'
            image_dir.mkdir(parents=True,exist_ok=True)
            label_dir.mkdir(parents=True,exist_ok=True)
            for image_name, xml_name in group:
                if xml_name not in items:
                    raise FileNotFoundError(f'XML missing for {image_name}')
                (image_dir / Path(image_name).name).write_bytes(z.read(image_name))
                (label_dir / (Path(image_name).stem+'.txt')).write_text(xml_to_yolo(z.read(xml_name)),encoding='utf-8')
            print(f'{split}: {len(group)} images')
    config = {'path':str(DATASET.resolve()),'train':'train/images','val':'val/images','test':'test/images','names':dict(enumerate(CLASSES))}
    yaml_path.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
    return yaml_path

if __name__ == '__main__':
    prepare_dataset()
