from scaaml.aes_forward import AESSBOX
import numpy as np
import h5py
import numpy as np
from Crypto.Cipher import AES
import tables

def ds_for_extracted_datasets(dataset):
  for dataset in datasets:
    f = h5py.File(data_path[dataset],'r')

    for i in range(int(len(list(f['Attack_traces']['traces']))/256)):
      k = i*256
      traces = np.array(f['Attack_traces']['traces'])[k:k+256]
      # cts = np.array(f['Attack_traces']['metadata']['ciphertext'])[k:k+256]
      pts = np.array(f['Attack_traces']['metadata']['plaintext'])[k:k+256]
      keys = np.array(f['Attack_traces']['metadata']['key'])[k:k+256]
      cts = []
      for text in range(len(pts)):
        cipher = AES.new(np.array(keys[text]).tobytes(), AES.MODE_ECB)
        cts.append(np.frombuffer(cipher.encrypt(np.array(pts[text]).tobytes()), dtype=np.uint8))
      sub_bytes_out = np.array([AESSBOX.sub_bytes_out(bytearray(f['Attack_traces']['metadata']['key'][k+m]), bytearray(f['Attack_traces']['metadata']['plaintext'][k+m])) for m in range(256)])
      sub_bytes_in = np.array([AESSBOX.sub_bytes_in(bytearray(f['Attack_traces']['metadata']['key'][k+m]), bytearray(f['Attack_traces']['metadata']['plaintext'][k+m])) for m in range(256)])

      a = list(traces.shape)
      a.append(1)
      traces = np.reshape(traces, tuple(a))
      keys = keys.T
      cts = np.array(cts).T
      pts = pts.T
      sub_bytes_out = sub_bytes_out.T
      sub_bytes_in = sub_bytes_in.T

      np.savez(f'./scaaml/scaaml_intro/datasets/{dataset}/test/{i}.npz',
        traces=traces,
        keys=keys,
        cts=cts,
        pts=pts,
        sub_bytes_out=sub_bytes_out,
        sub_bytes_in=sub_bytes_in)

      traces = np.array(f['Profiling_traces']['traces'])[k:k+256]
      # cts = np.array(f['Profiling_traces']['metadata']['ciphertext'])[k:k+256]
      pts = np.array(f['Profiling_traces']['metadata']['plaintext'])[k:k+256]
      keys = np.array(f['Profiling_traces']['metadata']['key'])[k:k+256]
      cts = []
      for text in range(len(pts)):
        cipher = AES.new(np.array(keys[text]).tobytes(), AES.MODE_ECB)
        cts.append(np.frombuffer(cipher.encrypt(np.array(pts[text]).tobytes()), dtype=np.uint8))
      sub_bytes_out = np.array([AESSBOX.sub_bytes_out(bytearray(f['Profiling_traces']['metadata']['key'][k+m]), bytearray(f['Profiling_traces']['metadata']['plaintext'][k+m])) for m in range(256)])
      sub_bytes_in = np.array([AESSBOX.sub_bytes_in(bytearray(f['Profiling_traces']['metadata']['key'][k+m]), bytearray(f['Profiling_traces']['metadata']['plaintext'][k+m])) for m in range(256)])

      a = list(traces.shape)
      a.append(1)
      traces = np.reshape(traces, tuple(a))
      keys = keys.T
      cts = np.array(cts).T
      pts = pts.T
      sub_bytes_out = sub_bytes_out.T
      sub_bytes_in = sub_bytes_in.T

      np.savez(f'./scaaml/scaaml_intro/datasets/{dataset}/train/{i}.npz',
        traces=traces,
        keys=keys,
        cts=cts,
        pts=pts,
        sub_bytes_out=sub_bytes_out,
        sub_bytes_in=sub_bytes_in)


def ds_for_raw_data(raw_data):
  h5file = tables.open_file(raw_data, mode="r")
  traces = np.array(h5file.root.traces)
  plaintext = np.array(h5file.root.metadata.cols.plaintext)
  key = np.array(h5file.root.metadata.cols.key)

  s = int(len(traces)/2)

  for i in range(0, s, 256):
    traces_256 = traces[i:i+256]
    plaintext_256 = plaintext[i:i+256]
    key_256 = key[i:i+256]
    sub_bytes_out = np.array([AESSBOX.sub_bytes_out(bytearray(key_256[m]), bytearray(plaintext_256[m])) for m in range(256)])
    sub_bytes_in = np.array([AESSBOX.sub_bytes_in(bytearray(key_256[m]), bytearray(plaintext_256[m])) for m in range(256)])
    cipher_256 = []
    for text in range(256):
      cipher = AES.new(np.array(key_256[text]).tobytes(), AES.MODE_ECB)
      cipher_256.append(np.frombuffer(cipher.encrypt(np.array(plaintext_256[text]).tobytes()), dtype=np.uint8))

    traces_256 = np.array(traces_256)
    a = list(traces_256.shape)
    a.append(1)
    traces_256 = np.reshape(traces_256, tuple(a))
    plaintext_256 = plaintext_256.T
    key_256 = key_256.T
    sub_bytes_out = sub_bytes_out.T
    sub_bytes_in = sub_bytes_in.T
    cipher_256 = np.array(cipher_256).T

    np.savez(f'./scaaml/scaaml_intro/datasets/ASCAD_raw/train/{int(i/256)}.npz',
      traces=traces_256,
      keys=key_256,
      cts=cipher_256,
      pts=plaintext_256,
      sub_bytes_out=sub_bytes_out,
      sub_bytes_in=sub_bytes_in)

    traces_256 = traces[i+s:i+s+256]
    plaintext_256 = plaintext[i+s:i+s+256]
    key_256 = key[i+s:i+s+256]
    sub_bytes_out = np.array([AESSBOX.sub_bytes_out(bytearray(key_256[m]), bytearray(plaintext_256[m])) for m in range(256)])
    sub_bytes_in = np.array([AESSBOX.sub_bytes_in(bytearray(key_256[m]), bytearray(plaintext_256[m])) for m in range(256)])
    cipher_256 = []
    for text in range(256):
      cipher = AES.new(np.array(key_256[text]).tobytes(), AES.MODE_ECB)
      cipher_256.append(np.frombuffer(cipher.encrypt(np.array(plaintext_256[text]).tobytes()), dtype=np.uint8))

    traces_256 = np.array(traces_256)
    a = list(traces_256.shape)
    a.append(1)
    traces_256 = np.reshape(traces_256, tuple(a))
    plaintext_256 = plaintext_256.T
    key_256 = key_256.T
    sub_bytes_out = sub_bytes_out.T
    sub_bytes_in = sub_bytes_in.T
    cipher_256 = np.array(cipher_256).T

    np.savez(f'./scaaml/scaaml_intro/datasets/ASCAD_raw/test/{int(i/256)}.npz',
      traces=traces_256,
      keys=key_256,
      cts=cipher_256,
      pts=plaintext_256,
      sub_bytes_out=sub_bytes_out,
      sub_bytes_in=sub_bytes_in)


if __name__ == "__main__":

  datasets = ["ASCAD", "ASCAD_desync50", "ASCAD_desync100", "ASCAD_var_key", "ASCAD_var_key_desync50", "ASCAD_var_key_desync100"]

  data_path = {
      "ASCAD": "ASCAD.h5",
      "ASCAD_desync50": "ASCAD_desync50.h5",
      "ASCAD_desync100": "ASCAD_desync100.h5",
      "ASCAD_var_key": "ASCAD-variable.h5",
      "ASCAD_var_key_desync50": "ascad-variable-desync50.h5",
      "ASCAD_var_key_desync100": "ascad-variable-desync100.h5"
  }

  raw_data = 'ATMega8515_raw_traces.h5'

  # ds_for_extracted_datasets(datasets)
  ds_for_raw_data(raw_data)

