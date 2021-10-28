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

    chip_id = 1
    comment = "this is a test"
    purpose = "train"
    example_per_shard = 1

    key = np.random.randint(0, 255, 16)
    key2 = np.random.randint(0, 255, 16)
    trace1 = np.random.rand(1024)

    ds = Dataset(root_path=root_path,
                 architecture=architecture,
                 implementation=implementation,
                 algorithm=algorithm,
                 version=version,
                 purpose=purpose,
                 comment=comment,
                 chip_id=chip_id,
                 examples_per_shard=example_per_shard,
                 measurements_info=minfo,
                 attack_points_info=apinfo)

    ds.new_shard(key, 1, 'train')
    ds.write_example({"key": key}, {"trace1": trace1})
    ds.close_shard()

    # 256 keys - with uniform bytes
    ds.new_shard(key2, 1, 'test')
    ds.write_example({"key": key2}, {"trace1": trace1})
    ds.close_shard()

    # check dataset integrity and consistency
    ds.check()
    slug = ds.slug
    # reload
    ds2 = Dataset.from_config(root_path / slug)
    ds2.inspect('train', 0, 1)
