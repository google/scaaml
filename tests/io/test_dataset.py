import numpy as np
import copy
import json
from pathlib import Path
import pytest
from unittest.mock import patch

from scaaml.io import Dataset
from scaaml.io.shard import Shard
from scaaml.io import utils as siutils


@patch.object(Path, 'read_text')
@patch.object(Shard, '__init__')
@patch.object(Shard, 'read')
def test_inspect(mock_shard_read, mock_shard_init, mock_read_text):
    split = 'test'
    shard_id = 0
    num_example = 5
    mock_shard_init.return_value = None
    config = {
        'shards_list': {
            'test': [{
                'path': 'test/0_abcd_1.tfrec'
            }, {
                'path': 'test/2_ef12_3.tfrec'
            }],
            'train': [{
                'path': 'train/2_dcea_0.tfrec'
            }, {
                'path': 'train/3_beef_1.tfrec'
            }],
        },
        'attack_points_info': {
            'ap_info': 'something'
        },
        'measurements_info': {
            'm_info': 'else'
        },
        'compression': 'GZIP',
    }
    mock_read_text.return_value = json.dumps(config)
    dir_dataset_ok = Path('/home/notanuser')

    x = Dataset.inspect(dir_dataset_ok,
                        split=split,
                        shard_id=shard_id,
                        num_example=num_example)

    shard_filename = str(dir_dataset_ok /
                         config['shards_list'][split][shard_id]['path'])
    mock_shard_init.assert_called_once_with(
        shard_filename,
        attack_points_info=config['attack_points_info'],
        measurements_info=config['measurements_info'],
        compression=config['compression'])
    mock_shard_read.assert_called_once_with(num=num_example)
    assert x == mock_shard_read.return_value


@patch.object(siutils, 'sha256sum')
def test_check_sha256sums(mock_sha256sum):
    shadict = {
        "test/0_01f6e272b933ec2b80ab53af245f7fa6_0.tfrec":
        "b1e3dc7c217b154ded5631d95d6265c6b1ad348ac4968acb1a74b9fb49c09c42",
        "test/1_f2e8de7fdbc602f96261ba5f8d182d73_0.tfrec":
        "7a9b214d76f68b4e1a9abf833314ae5909e96b6c4c9f81c7a020a63913dfc51c",
        "test/0_69a283f6b1eea6327afdb30f76e6fe30_0.tfrec":
        "f61009a4c6f5a77aa2c6da6d1882a50c3bd6345010966144d16e634ceeaeb730",
    }
    dpath = Path('/home/noanuser/notadir')
    mock_sha256sum.side_effect = lambda x: shadict[f'{x.parent.name}/{x.name}']
    shards_list = {
        'test': [{
            'path': f,
            'sha256': s,
        } for f, s in shadict.items()],
    }
    pbar = lambda *args, **kwargs: args[0]
    Dataset._check_sha256sums(shards_list=shards_list, dpath=dpath, pbar=pbar)

    mock_sha256sum.side_effect = lambda _: 'abcd'
    with pytest.raises(ValueError) as sha_error:
        Dataset._check_sha256sums(shards_list=shards_list,
                                  dpath=dpath,
                                  pbar=pbar)
    assert "SHA256 miss-match" in str(sha_error.value)


def test_check_metadata():
    config = {
        'shards_list': {
            'test': [{
                "examples": 64,
                "group": 0,
                "key": "01F6E272B933EC2B80AB53AF245F7FA6",
                "part": 0,
            }, {
                "examples": 64,
                "group": 1,
                "key": "F2E8DE7FDBC602F96261BA5F8D182D73",
                "part": 0,
            }, {
                "examples": 64,
                "group": 0,
                "key": "69A283F6B1EEA6327AFDB30F76E6FE30",
                "part": 0,
            }],
            'train': [
                {
                    "examples": 64,
                    "group": 0,
                    "key": "A4F6C39380E6D85CD2D4D5BD7EED11A8",
                    "part": 0,
                },
                {
                    "examples": 64,
                    "group": 0,
                    "key": "A4F6C39380E6D85CD2D4D5BD7EED11A8",
                    "part": 1,
                },
            ]
        },
        'examples_per_group': {
            "test": {
                "0": 2 * 64,
                "1": 1 * 64,
            },
            "train": {
                "0": 2 * 64
            }
        },
        'examples_per_split': {
            "test": 3 * 64,
            "train": 2 * 64
        },
        'examples_per_shard': 64,
    }
    Dataset._check_metadata(config=config)

    bad_config = copy.deepcopy(config)
    bad_config['examples_per_split']['test'] = 5
    with pytest.raises(ValueError) as metadata_error:
        Dataset._check_metadata(config=bad_config)
    assert "Num shards in shard_list !=" in str(metadata_error.value)


def test_shallow_check():
    pbar = lambda *args, **kwargs: args[0]
    seen_keys = set()
    train_shards = [
        {
            "key": "FFE8"
        },
    ]
    Dataset._shallow_check(seen_keys, train_shards, pbar)

    seen_keys.add(np.array([255, 232], dtype=np.uint8).tobytes())
    with pytest.raises(ValueError) as intersection_error:
        Dataset._shallow_check(seen_keys, train_shards, pbar)
    assert 'Duplicate key' in str(intersection_error.value)


@patch.object(Dataset, 'inspect')
def test_deep_check(mock_inspect):
    mock_inspect.return_value.as_numpy_iterator.return_value = (
        {
            'key': np.array([0, 1, 2, 255])
        },
        {
            'key': np.array([3, 1, 4, 1])
        },
    )
    seen_keys = set()
    pbar = lambda *args, **kwargs: args[0]
    train_shards = [
        {},
    ]
    Dataset._deep_check(seen_keys=seen_keys,
                        dpath='/home/notanuser/notdir',
                        train_shards=train_shards,
                        examples_per_shard=64,
                        pbar=pbar)

    seen_keys.add(np.array([3, 1, 4, 1], dtype=np.uint8).tobytes())
    with pytest.raises(ValueError) as intersection_error:
        Dataset._deep_check(seen_keys=seen_keys,
                            dpath='/home/notanuser/notdir',
                            train_shards=train_shards,
                            examples_per_shard=64,
                            pbar=pbar)
    assert 'Duplicate key' in str(intersection_error.value)


def test_basic_workflow(tmp_path):
    root_path = tmp_path
    architecture = 'arch'
    implementation = 'implem'
    algorithm = 'algo'
    version = 1
    minfo = {
        # test missing measurem,net raise value
        # test extra measurement raise value
        "trace1": {
            "type": "power",
            "len": 1024,
        }
    }
    apinfo = {
        "key": {
            "len": 16,
            "max_val": 256
        },

        # test missing attack point raise value
        # test extra attack point raise value
        # "sub_byte_in": {
        #     "len": 16,
        #     "max_val": 256
        # }
    }

    shortname = "ARCHALGO21"
    description = "this is a test"
    url = "https://"
    example_per_shard = 1
    fw_sha256 = "A2424512D"
    key = np.random.randint(0, 255, 16)
    key2 = np.random.randint(0, 255, 16)
    trace1 = np.random.rand(1024)

    ds = Dataset(root_path=root_path,
                 shortname=shortname,
                 architecture=architecture,
                 implementation=implementation,
                 algorithm=algorithm,
                 version=version,
                 description=description,
                 url=url,
                 firmware_sha256=fw_sha256,
                 examples_per_shard=example_per_shard,
                 measurements_info=minfo,
                 attack_points_info=apinfo)

    chip_id = 1
    ds.new_shard(key=key, part=0, split='train', group=0, chip_id=chip_id)
    ds.write_example({"key": key}, {"trace1": trace1})
    ds.close_shard()

    # 256 keys - with uniform bytes

    ds.new_shard(key=key2, part=1, split='train', group=0, chip_id=chip_id)
    ds.write_example({"key": key2}, {"trace1": trace1})
    ds.close_shard()

    # check dataset integrity and consistency
    ds.check()
    slug = ds.slug
    # reload
    ds2 = Dataset.from_config(root_path / slug)
    ds2.inspect(root_path / slug, 'train', 0, 1)
    ds2.summary(root_path / slug)


def test_cleanup_shards(tmp_path):
    def shard_info(group: int, key: str, part: int):
        return {
            'path': Dataset._shard_name(shard_group=group,
                                        shard_key=key,
                                        shard_part=part),
            'examples': 64,
            'size': 811345,
            'sha256': 'beef',
            'group': group,
            'key': key,
            'part': part,
            'chip_id': 13,
        }  # yapf: disable

    old_config = {  # Some fields omitted.
        'examples_per_shard': 64,
        'examples_per_group': {
            'test': {
                '0': 2 * 64,
                '1': 1 * 64,
                '2': 1 * 64,
                '3': 2 * 64,
            },
            'train': {
                '0': 5 * 64,
            }
        },
        'examples_per_split': {
            'test': 6 * 64,
            'train': 5 * 64,
        },
        'keys_per_split': {
            'test': 4,
            'train': 4,
        },
        'keys_per_group': {
            'test': {
                '0': 1,
                '1': 1,
                '2': 1,
                '3': 1,
            },
            'train': {
                '0': 4,
            }
        },
        'shards_list': {
            'test': [
                shard_info(group=0, key='KEYA', part=0),
                shard_info(group=0, key='KEYA', part=2),  # del
                shard_info(group=1, key='KEYB', part=2),  # del
                shard_info(group=2, key='KEYC', part=2),
                shard_info(group=3, key='KEYD', part=1),
                shard_info(group=3, key='KEYD', part=2),
            ],
            'train': [
                shard_info(group=0, key='keyA', part=2),  # del
                shard_info(group=0, key='keyA', part=1),
                shard_info(group=0, key='keyB', part=3),
                shard_info(group=0, key='keyC', part=4),  # del
                shard_info(group=0, key='keyD', part=5),
            ],
        },
    }
    # Populate the mock database
    Dataset._get_config_path(tmp_path).write_text(json.dumps(old_config))
    for s in ['train', 'test', 'holdout']:
        (tmp_path / s).mkdir()

    new_config = copy.deepcopy(old_config)
    # Delete some files. Remember to update 'examples_per_group' and 'example_per_shard'.
    for i in sorted([0, 3], reverse=True):  # Remove in descending order.
        del new_config['shards_list']['train'][i]
    new_config['examples_per_group']['train'] = {
        '0': 3 * 64,
    }
    new_config['examples_per_split']['train'] = 3 * 64
    new_config['keys_per_split']['train'] = 3
    new_config['keys_per_group']['train'] = {
        '0': 3,
    }
    for i in sorted([1, 2], reverse=True):  # Remove in descending order.
        del new_config['shards_list']['test'][i]
    new_config['examples_per_group']['test'] = {
        '0': 1 * 64,
        '1': 0 * 64,
        '2': 1 * 64,
        '3': 2 * 64,
    }
    new_config['examples_per_split']['test'] = 4 * 64
    new_config['keys_per_split']['test'] = 3
    new_config['keys_per_group']['test'] = {
        '0': 1,
        '1': 0,
        '2': 1,
        '3': 1,
    }
    for i in []:  # Fill this split in old_config first
        del new_config['shards_list']['holdout'][i]
    # Create existing files.
    for s in new_config['shards_list']:
        for f in new_config['shards_list'][s]:
            (tmp_path / f['path']).touch()
    # Other files should be neither deleted nor added to the dataset.
    other_files = [
        tmp_path / 'iamnothere.txt',
        tmp_path / 'train' / 'not_a_shard.tfrec',
    ]
    for f in other_files:
        f.touch()

    corrected_config = Dataset._cleanup_shards(tmp_path, print_info=False)

    # New config is ok.
    # Loop for better readability.
    for k in corrected_config:
        assert corrected_config[k] == new_config[k], f'{k} is different'
    # Test all (so far tested that corrected_config is a subset of new_config).
    assert corrected_config == new_config
    # Other files still present.
    for f in other_files:
        assert f.exists()


def test_shard_info_from_name():
    assert Dataset._shard_info_from_name(
        '1_c80b174b5ce880a3557db2152598cafe_2.tfrec') == {
            'shard_group': 1,
            'shard_key': 'c80b174b5ce880a3557db2152598cafe',
            'shard_part': 2,
        }


def test_shard_info_from_name_directory():
    assert Dataset._shard_info_from_name(
        'some/directory/1_c80b174b5ce880a3557db2152598cafe_2.tfrec') == {
            'shard_group': 1,
            'shard_key': 'c80b174b5ce880a3557db2152598cafe',
            'shard_part': 2,
        }
    assert Dataset._shard_info_from_name(
        'win_dir\\1_c80b174b5ce880a3557db2152598cafe_2.tfrec') == {
            'shard_group': 1,
            'shard_key': 'c80b174b5ce880a3557db2152598cafe',
            'shard_part': 2,
        }


def test_shard_name():
    assert Dataset._shard_name(
        shard_group=1,
        shard_key='c80b174b5ce880a3557db2152598cafe',
        shard_part=2) == '1_c80b174b5ce880a3557db2152598cafe_2.tfrec'


def test_shard_info_from_name_identity():
    tests = [
        {
            'shard_group': 1,
            'shard_key': 'cafe',
            'shard_part': 0,
        },
        {
            'shard_group': 0,
            'shard_key': 'dead',
            'shard_part': 1,
        },
        {
            'shard_group': 2,
            'shard_key': 'beef',
            'shard_part': 4,
        },
        {
            'shard_group': 3,
            'shard_key': '0123',
            'shard_part': 2,
        },
        {
            'shard_group': 4,
            'shard_key': 'c0de',
            'shard_part': 3,
        },
    ]
    for t in tests:
        assert Dataset._shard_info_from_name(Dataset._shard_name(**t)) == t
