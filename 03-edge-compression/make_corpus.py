"""Build corpus.bin: the concatenated Python standard-library source of the running interpreter (~4.6 MB)."""
import glob, sysconfig
files = sorted(glob.glob(sysconfig.get_paths()["stdlib"] + "/*.py"))
data = b"".join(open(f, "rb").read() for f in files)
open("corpus.bin", "wb").write(data)
print(f"{len(files)} files, {len(data):,} bytes -> corpus.bin")
