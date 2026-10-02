import wfdb
import matplotlib.pyplot as plt

data_path = r"C:\Users\hp\Downloads\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\100"
record = wfdb.rdrecord(data_path)
annotation = wfdb.rdann(data_path, "atr")

print("Signal shape:", record.p_signal.shape)
print("Sampling frequency:", record.fs)
print("Number of annotations:", len(annotation.sample))

wfdb.plot_wfdb(
    record=record,
    annotation=annotation,
    time_units="seconds"
)

plt.show()