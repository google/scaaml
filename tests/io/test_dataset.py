import numpy as np
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
