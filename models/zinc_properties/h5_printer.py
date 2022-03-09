from silx.io.dictdump import h5todict

file = "zinc_prop_pred.h5"
in_file = h5todict(file)
print(in_file)

