import numpy as np
import copy
import json

from scaaml.io import Dataset


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
